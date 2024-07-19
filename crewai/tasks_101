Here's a markdown cheatsheet for CrewAI tasks, focusing on how to build and customize tasks for your bots based on the provided information:

# CrewAI Tasks Cheatsheet: Building and Customizing Tasks

## Core Concepts

- **Task**: A specific assignment for agents that includes details like description, responsible agent, required tools, and expected output.
- **Agent**: The entity assigned to complete the task.
- **Crew**: A group of agents working together, potentially on collaborative tasks.

## Task Attributes

| Attribute          | Parameters         | Description                                                   |
|--------------------|--------------------|---------------------------------------------------------------|
| `description`      | `str`              | A clear statement of what the task entails.                  |
| `agent`            | `Agent`            | The agent responsible for completing the task.                |
| `expected_output`  | `str`              | Description of what the task's completion looks like.         |
| `tools`            | `List[Tool]`       | Optional tools the agent can use to perform the task.        |
| `async_execution`  | `bool`             | If set, the task executes asynchronously.                     |
| `context`          | `List[Task]`       | Tasks whose outputs are used as context for this task.       |
| `config`           | `dict`             | Additional configuration details for the agent.              |
| `output_json`      | `bool`             | Outputs a JSON object.                                       |
| `output_pydantic`  | `bool`             | Outputs a Pydantic model object.                             |
| `output_file`      | `str`              | Saves the task output to a specified file.                  |
| `callback`         | `callable`         | A function executed with the task's output upon completion.  |
| `human_input`      | `bool`             | Indicates if human feedback is required at the end.          |

## Creating a Task

To create a task, define its scope, responsible agent, and any additional attributes:

```python
from crewai import Task

task = Task(
    description='Find and summarize the latest and most relevant news on AI',
    agent=sales_agent,
    expected_output='A bullet list summary of the top 5 most important AI news'
)
```

## Integrating Tools with Tasks

You can enhance task performance by integrating tools:

```python
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

# Set up the agent
research_agent = Agent(
    role='Researcher',
    goal='Find and summarize the latest AI news',
    backstory="You're a researcher at a large company responsible for analyzing data and providing insights.",
    verbose=True
)

# Set up the tool
search_tool = SerperDevTool()

# Create the task
task = Task(
    description='Find and summarize the latest AI news',
    expected_output='A bullet list summary of the top 5 most important AI news',
    agent=research_agent,
    tools=[search_tool]
)

# Create a crew and execute the task
crew = Crew(agents=[research_agent], tasks=[task], verbose=2)
result = crew.kickoff()
print(result)
```

## Referring to Other Tasks

You can specify that a task uses the output from other tasks as context:

```python
# Define tasks
research_ai_task = Task(
    description='Find and summarize the latest AI news',
    expected_output='A bullet list summary of the top 5 most important AI news',
    agent=research_agent,
    tools=[search_tool],
    async_execution=True
)

research_ops_task = Task(
    description='Find and summarize the latest AI Ops news',
    expected_output='A bullet list summary of the top 5 most important AI Ops news',
    agent=research_agent,
    tools=[search_tool],
    async_execution=True
)

write_blog_task = Task(
    description="Write a full blog post about the importance of AI and its latest news",
    expected_output='Full blog post that is 4 paragraphs long',
    agent=writer_agent,
    context=[research_ai_task, research_ops_task]
)
```

## Asynchronous Execution

Tasks can be executed asynchronously, allowing the crew to continue working on other tasks:

```python
list_ideas = Task(
    description="List of 5 interesting ideas to explore for an article about AI.",
    expected_output="Bullet point list of 5 ideas for an article.",
    agent=researcher,
    async_execution=True
)

list_important_history = Task(
    description="Research the history of AI and give me the 5 most important events.",
    expected_output="Bullet point list of 5 important events.",
    agent=researcher,
    async_execution=True
)

write_article = Task(
    description="Write an article about AI, its history, and interesting ideas.",
    expected_output="A 4 paragraph article about AI.",
    agent=writer,
    context=[list_ideas, list_important_history]
)
```

## Callback Mechanism

You can define a callback function that executes after the task is completed:

```python
def callback_function(output):
    print(f"Task completed! Output: {output.raw_output}")

research_task = Task(
    description='Find and summarize the latest AI news',
    expected_output='A bullet list summary of the top 5 most important AI news',
    agent=research_agent,
    tools=[search_tool],
    callback=callback_function
)
```

## Accessing Task Output

After the crew runs, you can access the output of a specific task:

```python
task1 = Task(
    description='Find and summarize the latest AI news',
    expected_output='A bullet list summary of the top 5 most important AI news',
    agent=research_agent,
    tools=[search_tool]
)

crew = Crew(agents=[research_agent], tasks=[task1], verbose=2)
result = crew.kickoff()

print(f"Task completed! Output: {task1.output.raw_output}")
```

## Tool Override Mechanism

Specifying tools in a task allows for dynamic adaptation of agent capabilities, enhancing flexibility.

## Error Handling and Validation

CrewAI includes validation mechanisms to ensure task attributes are correctly set, such as:

- Ensuring only one output type is set per task.
- Preventing manual assignment of the `id` attribute.

## Example: Creating a Task for a Research Agent

Here's a complete example of creating a research task with tools:

```python
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

# Set up the agent
research_agent = Agent(
    role='Researcher',
    goal='Gather and summarize AI news',
    backstory="You're an expert researcher focused on AI developments.",
    verbose=True
)

# Set up the tool
search_tool = SerperDevTool()

# Create the task
task = Task(
    description='Find and summarize the latest AI news',
    expected_output='A bullet list summary of the top 5 most important AI news',
    agent=research_agent,
    tools=[search_tool],
    async_execution=False
)

# Create and run the crew
crew = Crew(agents=[research_agent], tasks=[task], verbose=2)
result = crew.kickoff()
print(result)
```

This cheatsheet provides a quick reference for building and customizing tasks in CrewAI, allowing you to enhance your bots' capabilities effectively. Adjust the examples to fit your specific use case and requirements.

Citations:
[1] https://docs.crewai.com/core-concepts/Tasks/
