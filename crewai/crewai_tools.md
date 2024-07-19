Here's a cheatsheet for CrewAI tools, focusing on how to add and find tools for your bot:

# CrewAI Tools Cheatsheet: Empowering Your Agents

## Core Concepts

- **Tool**: A skill or function that agents can use to perform actions.
- **crewAI Toolkit**: A collection of pre-built tools for common tasks.
- **LangChain Tools**: Compatible tools from the LangChain ecosystem.

## Installing CrewAI Tools

To access the full range of CrewAI tools:

```bash
pip install 'crewai[tools]'
```

## Adding Tools to Agents

Tools are added when creating an Agent:

```python
from crewai import Agent
from crewai_tools import SerperDevTool, WebsiteSearchTool

agent = Agent(
    role="Researcher",
    goal="Gather information on AI trends",
    backstory="You're an AI expert always seeking the latest developments.",
    tools=[SerperDevTool(), WebsiteSearchTool()],
    verbose=True
)
```

## Available CrewAI Tools

CrewAI offers a wide range of tools. Here are some key ones:

1. **Web and Search Tools**:
   - `SerperDevTool`: For web searches
   - `WebsiteSearchTool`: For searching specific websites
   - `ScrapeWebsiteTool`: For web scraping

2. **File and Data Tools**:
   - `DirectoryReadTool`: For reading directory contents
   - `FileReadTool`: For reading various file formats
   - `CSVSearchTool`: For searching CSV files
   - `PDFSearchTool`: For searching PDF documents

3. **Code and Documentation Tools**:
   - `CodeDocsSearchTool`: For searching code documentation
   - `GithubSearchTool`: For searching GitHub repositories

4. **Database Tools**:
   - `PGSearchTool`: For PostgreSQL database searches

5. **Media Tools**:
   - `YoutubeChannelSearchTool`: For searching YouTube channels
   - `YoutubeVideoSearchTool`: For searching within YouTube videos

## Creating Custom Tools

You can create custom tools in two ways:

1. Subclassing `BaseTool`:

```python
from crewai_tools import BaseTool

class MyCustomTool(BaseTool):
    name: str = "My Custom Tool"
    description: str = "Description of what this tool does"

    def _run(self, argument: str) -> str:
        # Implement your tool logic here
        return "Result from custom tool"
```

2. Using the `@tool` decorator:

```python
from crewai_tools import tool

@tool("My Custom Tool")
def my_tool(argument: str) -> str:
    """Description of what this tool does"""
    # Implement your tool logic here
    return "Result from custom tool"
```

## Finding Tools for Your Bot

1. **CrewAI Documentation**: Check the [official CrewAI tools documentation](https://docs.crewai.com/core-concepts/Tools/) for the latest list of available tools.

2. **LangChain Tools**: CrewAI is compatible with LangChain tools. Explore the [LangChain tools documentation](https://python.langchain.com/docs/modules/agents/tools/) for additional options.

3. **Community Resources**: Look for community-created tools on platforms like GitHub or PyPI.

4. **Custom Development**: Create your own tools based on your specific requirements.

## Best Practices for Using Tools

1. **Choose Relevant Tools**: Select tools that align with your agent's role and goals.
2. **Combine Tools**: Use multiple tools to create more capable agents.
3. **Error Handling**: Leverage built-in error handling in CrewAI tools.
4. **Caching**: Utilize tool caching to improve performance.
5. **Custom Caching**: Implement custom caching logic for fine-grained control:

```python
@tool
def my_tool(argument: str) -> str:
    """Tool description"""
    return "Result"

def cache_func(args, result):
    # Define custom caching logic
    return should_cache

my_tool.cache_function = cache_func
```

6. **Tool Arguments**: Be aware that tools can accept various argument types, not just strings.

## Example: Creating a Research Agent with Tools

```python
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, WebsiteSearchTool

# Set up tools
search_tool = SerperDevTool()
web_tool = WebsiteSearchTool()

# Create an agent with tools
researcher = Agent(
    role="AI Trend Researcher",
    goal="Provide cutting-edge insights on AI developments",
    backstory="You're a renowned AI analyst with a knack for spotting trends.",
    tools=[search_tool, web_tool],
    verbose=True
)

# Create a task for the agent
research_task = Task(
    description="Research the latest breakthroughs in natural language processing",
    agent=researcher
)

# Create and run a crew
crew = Crew(agents=[researcher], tasks=[research_task])
result = crew.kickoff()
print(result)
```

This cheatsheet provides a quick reference for adding and using tools in CrewAI, allowing you to enhance your bot's capabilities. Remember to check the official documentation for the most up-to-date information on available tools and best practices[1].

Citations:
[1] https://docs.crewai.com/core-concepts/Tools/
