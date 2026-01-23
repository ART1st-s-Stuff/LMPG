from typing import List, Dict, Optional
import uuid
import json
from pathlib import Path
from threading import Thread

from utils.context import DefaultContextType

from flask import Flask, render_template, jsonify

# 获取当前文件所在目录
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / 'templates'

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

# .trace 文件夹在项目根目录
# 从 trace/trace.py 到项目根目录需要向上一级
PROJECT_ROOT = BASE_DIR.parent
TRACE_DIR = PROJECT_ROOT / '.trace'
TRACE_DIR.mkdir(exist_ok=True)

class Trace(List[DefaultContextType]):
    def __init__(self, trace_id: str):
        super().__init__()
        self.trace_id = trace_id
    
    def add(self, item: DefaultContextType):
        self.append(item)
        self._save()

    @property
    def create_at(self) -> float:
        trace_file = TRACE_DIR / f"{self.trace_id}.json"
        return trace_file.stat().st_mtime
    
    def _save(self):
        """保存 trace 到文件"""
        trace_file = TRACE_DIR / f"{self.trace_id}.json"
        with open(trace_file, 'w', encoding='utf-8') as f:
            json.dump(list(self), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, trace_id: str) -> 'Trace':
        """从文件加载 trace"""
        trace_file = TRACE_DIR / f"{trace_id}.json"
        if not trace_file.exists():
            raise FileNotFoundError(f"Trace {trace_id} not found")
        
        trace = cls(trace_id)
        with open(trace_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            trace.extend(data)
        return trace

class TraceManager:
    def __init__(self):
        self.traces = {}
        self._load_all_traces()
    
    def _load_all_traces(self):
        """从 .trace 文件夹加载所有 traces"""
        if not TRACE_DIR.exists():
            return
        
        for trace_file in TRACE_DIR.glob("*.json"):
            trace_id = trace_file.stem
            try:
                self.traces[trace_id] = Trace.load(trace_id)
            except Exception as e:
                print(f"Error loading trace {trace_id}: {e}")
    
    def new_trace(self) -> Trace:
        trace_id = str(uuid.uuid4())
        trace = Trace(trace_id)
        self.traces[trace_id] = trace
        trace._save()  # 创建空文件
        return trace
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """获取指定的 trace"""
        if trace_id in self.traces:
            return self.traces[trace_id]
        return None
    
    def list_traces(self) -> List[Dict]:
        """列出所有 traces 的元信息"""
        traces_info = []
        for trace_id, trace in self.traces.items():
            traces_info.append({
                'id': trace_id,
                'length': len(trace),
                'created_at': trace.create_at
            })
        # 按创建时间倒序排列
        traces_info.sort(key=lambda x: x['created_at'], reverse=True)
        return traces_info

# 全局 TraceManager 实例
trace_manager = TraceManager()

# Flask 路由
@app.route('/')
def index():
    """显示所有 traces 列表"""
    traces = trace_manager.list_traces()
    return render_template('traces_list.html', traces=traces)

@app.route('/trace/<trace_id>')
def view_trace(trace_id: str):
    """显示单个 trace 的详情"""
    trace = trace_manager.get_trace(trace_id)
    if trace is None:
        return f"Trace {trace_id} not found", 404
    
    # 计算统计信息
    stats = {
        'total': len(trace),
        'system': sum(1 for msg in trace if msg.get('role') == 'system'),
        'user': sum(1 for msg in trace if msg.get('role') == 'user'),
        'assistant': sum(1 for msg in trace if msg.get('role') == 'assistant'),
    }
    
    return render_template('trace_detail.html', trace=trace, trace_id=trace_id, stats=stats)

@app.route('/api/traces')
def api_traces():
    """API: 获取所有 traces 列表"""
    traces = trace_manager.list_traces()  # 每次 API 调用都刷新
    return jsonify(traces)

@app.route('/api/trace/<trace_id>')
def api_trace(trace_id: str):
    """API: 获取单个 trace 的详情"""
    trace = trace_manager.get_trace(trace_id)
    if trace is None:
        return jsonify({'error': 'Trace not found'}), 404
    
    return jsonify({
        'id': trace_id,
        'messages': list(trace)
    })

def start_server():
    trace_thread = Thread(target=lambda: app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000), daemon=True)
    trace_thread.start()
    print("Trace server started.")

start_server()