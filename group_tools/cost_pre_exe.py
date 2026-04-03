import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import json

def _load_costs(file_path):
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def build_exec_costs_everylevel(file_path):
    """
    计算规则：在每个agent的costs.agent 增加一个execution_cost字段，表示该agent的执行成本。
    该字段的计算规则，从叶子节点计算起；具体公式如下：
        level(n-1) = whole_cost(n-1) - advise_cost(n-1) - hire_cost(n-1) + Σ execution_cost(child_i(n))
    """
    agent_tree = _load_costs(file_path)
    # 写啊
    def compute_execution_costs(agent_manager):
        agents = agent_manager.get("agents", [])
        for agent in agents:
            child_manager = agent.get("child_manager")
            if child_manager:
                compute_execution_costs(child_manager)
            # 计算当前agent的execution_cost
            costs = agent.get("costs", {}).get("agent", {})
            whole_cost = costs.get("whole_cost", 0)
            advise_cost = costs.get("advise_cost", 0)
            hire_cost = costs.get("hire_cost", 0)
            system_prompt_size = costs.get("system_prompt_size", 0)
            # 计算子节点的execution_cost总和
            child_execution_cost_sum = 0
            if child_manager:
                for child_agent in child_manager.get("agents", []):
                    child_costs = child_agent.get("costs", {}).get("agent", {})
                    child_execution_cost = child_costs.get("execution_cost", 0)
                    child_execution_cost_sum += child_execution_cost
            execution_cost = (whole_cost - advise_cost - hire_cost + child_execution_cost_sum)
            costs["execution_cost"] = execution_cost
    compute_execution_costs(agent_tree)
    exec_costs_everylevel = agent_tree
    return exec_costs_everylevel

if __name__ == "__main__":
    file_path = os.path.join(project_root, "logs/resolve/puzzle_40/deepseek/agent_tree_40.json")
    exec_costs_everylevel = build_exec_costs_everylevel(file_path)
    import json
    print(json.dumps(exec_costs_everylevel, indent=2, ensure_ascii=False))