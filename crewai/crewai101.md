Here's a markdown cheatsheet for CrewAI, focusing on how to build agents based on the provided information:

# CrewAI Cheatsheet: Building Agents for Collaborative AI Systems

## Core Concepts

- **Agent**: An autonomous unit programmed to perform tasks, make decisions, and communicate with other agents.
- **Crew**: A team of agents working together towards a common goal.
- **Task**: A specific job assigned to an agent within a crew.

## Agent Creation

To create an agent in CrewAI:

```python
from crewai import Agent

agent = Agent(
    role="Data Analyst",
    goal="Extract actionable insights",
    backstory="You're a data analyst at a large company...",
    tools=[my_tool1, my_tool2],  # Optional
    llm=my_llm,  # Optional
    verbose=True  # Optional
)
```

## Key Agent Attributes

| Attribute | Description |
|-----------|-------------|
| `role` | Defines the agent's function within the crew |
| `goal` | The individual objective the agent aims to achieve |
| `backstory` | Provides context to the agent's role and goal |
| `tools` | Set of capabilities or functions the agent can use |
| `llm` | Language model that will run the agent |

## Optional Agent Parameters

- `function_calling_llm`: Specifies the LLM for tool calling
- `max_iter`: Maximum iterations before forced answer (default: 25)
- `max_rpm`: Maximum requests per minute to avoid rate limits
- `max_execution_time`: Maximum execution time for a task
- `verbose`: Enables detailed execution logs (default: False)
- `allow_delegation`: Enables task delegation to other agents (default: True)
- `step_callback`: Function called after each agent step
- `cache`: Enables caching for tool usage (default: True)

## Customizing Agent Templates

You can customize system, prompt, and response templates:

```python
agent = Agent(
    role="{topic} specialist",
    goal="Figure {goal} out",
    backstory="I am the master of {role}",
    system_template="<|start_header_id|>system<|end_header_id|>{{ .System }}<|eot_id|>",
    prompt_template="<|start_header_id|>user<|end_header_id|>{{ .Prompt }}<|eot_id|>",
    response_template="<|start_header_id|>assistant<|end_header_id|>{{ .Response }}<|eot_id|>"
)
```

## Integrating Third-Party Agents

CrewAI supports integration with third-party agents:

```python
from crewai import Agent, Task, Crew
from custom_agent import CustomAgent  # Extend CrewAI's BaseAgent

agent1 = CustomAgent(
    role="backstory agent",
    goal="who is {input}?",
    backstory="agent backstory",
    verbose=True
)

task1 = Task(
    expected_output="a short biography of {input}",
    description="a short biography of {input}",
    agent=agent1
)

# Create more agents and tasks as needed

my_crew = Crew(agents=[agent1, agent2], tasks=[task1, task2])
result = my_crew.kickoff(inputs={"input": "Mark Twain"})
```

## Best Practices for Building Agents

1. **Define Clear Roles**: Assign specific, well-defined roles to each agent.
2. **Set Precise Goals**: Ensure each agent has a clear, achievable goal.
3. **Provide Rich Backstories**: Use backstories to give context and depth to agents.
4. **Choose Appropriate Tools**: Equip agents with relevant tools for their tasks.
5. **Optimize Performance**: Use `max_iter`, `max_rpm`, and `max_execution_time` to control agent behavior.
6. **Enable Delegation**: Use `allow_delegation=True` for complex tasks requiring collaboration.
7. **Implement Logging**: Use `verbose=True` and `step_callback` for debugging and monitoring.
8. **Customize Templates**: Tailor system, prompt, and response templates for specific use cases.
9. **Integrate Third-Party Agents**: Extend BaseAgent class to incorporate external agent frameworks.
10. **Balance Autonomy and Control**: Design agents to be autonomous while maintaining overall control through the Crew structure.

## Example: Creating a Research Crew

```python
from crewai import Agent, Task, Crew

# Create Agents
researcher = Agent(
    role="Research Analyst",
    goal="Gather comprehensive information on AI trends",
    backstory="You're an expert in AI with a keen eye for emerging trends.",
    tools=[search_tool, database_tool]
)

writer = Agent(
    role="Technical Writer",
    goal="Synthesize research into a clear, concise report",
    backstory="You're skilled at translating complex information into readable content."
)

# Create Tasks
research_task = Task(
    description="Research the latest AI trends in natural language processing",
    agent=researcher
)

writing_task = Task(
    description="Write a 500-word summary report on the research findings",
    agent=writer,
    context=[research_task]
)

# Assemble the Crew
ai_trend_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task]
)

# Execute the Crew's mission
result = ai_trend_crew.kickoff()
print(result)
```

This cheatsheet provides a quick reference for creating and using agents in CrewAI, allowing for the development of sophisticated, collaborative AI systems. Customize the examples to fit your specific use case and requirements.

Citations:
[1] https://docs.crewai.com/core-concepts/Agents/
