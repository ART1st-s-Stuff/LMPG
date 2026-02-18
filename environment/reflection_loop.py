from utils.environment import Environment
from utils.tool import Toolset
import types

class TriggerReflection(Toolset):
    def __init__(self, reflection_prompt: str, force_trigger_rounds: int):
        super().__init__()
        self.prompt = reflection_prompt
        self.force_trigger_rounds = force_trigger_rounds
        self.current_round = 0
    
    def step(self):
        self.current_round += 1
    
    def reset(self):
        self.current_round = 0
        
    def should_reflect(self):
        return self.current_round == self.force_trigger_rounds
        
    def process_input(self, input: str) -> str:
        return input + "\n\n" + self.prompt
    
    @Toolset.structurized_tool(tool_name="trigger_reflection")
    def trigger_reflection(self):
        return self.prompt


def inject_reflection_loop(environment: Environment, reflection_prompt: str, force_trigger_rounds: int):
    tool = TriggerReflection(reflection_prompt, force_trigger_rounds)
    environment.tools["reflection_trigger"] = tool
    def reflection_looped_run(self, agent):
        agent.clear_state()
        for i in range(self.max_steps):
            if i == 0:
                agent_input = list(self.prompt.values())[0]
            elif tool.should_reflect():
                agent_input = tool.process_input(result)
            else:
                agent_input = result
            result = agent.step(agent_input)
            tool.step()
            self.num_steps += 1
            if self.stop_criteria(result):
                break
        return result
    environment.run = types.MethodType(reflection_looped_run, environment)
    return environment