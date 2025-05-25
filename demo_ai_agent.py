from transformers import pipeline
from typing import Dict, Callable
import re

# ===== Configuration =====
#MODEL_NAME = "google/flan-t5-small"
#MODEL_NAME = "google/flan-t5-base"
MODEL_NAME = "google/flan-t5-large"
LLM_SETTINGS = {
    "task": "text2text-generation",
    "model": MODEL_NAME,
    "max_new_tokens": 5
}

# ===== Agent Class =====
class Agent:
    """Main AI agent that routes prompts to appropriate tools"""
    def __init__(self):
        self.llm = pipeline(**LLM_SETTINGS)
        self.tools = self._initialize_tools()

    def run(self, prompt: str) -> str:
        """Process user input and route to appropriate tool"""
        tool_name = self._choose_tool(prompt)
        
        if not tool_name:
            return "Error: Could not select a tool. Try 'search [query]' or 'add X Y'"
        
        return self.tools[tool_name](prompt)

    def _initialize_tools(self) -> Dict[str, Callable]:
        """Register all available tools"""
        tools = {}

        @self._tool(tools)
        def search(query: str) -> str:
            """Search for information online (mocked version)"""
            return f"Search results for: {query}"

        @self._tool(tools)
        def add_numbers(prompt: str) -> str:
            """
            Extract numbers from prompt and return their sum
            Example: "Add 5 and 3" → "Result: 8.0"
            """
            numbers = re.findall(r"\d+", prompt)
            if len(numbers) < 2:
                return "Error: Please provide at least two numbers to add"
            
            total = 0.0
            for num in numbers:
                total += float(num)
            return f"Result: {total}"

        return tools

    def _tool(self, tools: Dict[str, Callable]) -> Callable:
        """Decorator to register functions as tools"""
        def decorator(func: Callable) -> Callable:
            tools[func.__name__] = func
            return func
        return decorator

    def _choose_tool(self, prompt: str) -> str:
        """Select tool using LLM with keyword fallback"""
        tool_name = self._try_llm_selection(prompt)
        return tool_name if tool_name else self._try_keyword_fallback(prompt)

    def _try_llm_selection(self, prompt: str) -> str:
        """Use LLM to select the appropriate tool"""
        #tool_list = list(self.tools.keys())
        """
        examples = "\n".join([
            f"- Task: 'Find cats' → search",
            f"- Task: 'Add 2+2' → add_numbers", 
            f"- Task: 'Tell time' -> none", 
            f"- Task: Sing a song' -> none"

        ])
        """
        
        # The "flan-t5-large" model fails for the negative case ("Tell a joke")
        # with the below prompt
        #llm_prompt = f"""Select one tool from {tool_list}. Examples:
        #{examples}
        #Task: "{prompt}"
        #Respond ONLY with the tool name: """
        
        # The "flan-t5-large" model works with this explicit prompt
        llm_prompt = f"""Classify the task. Respond ONLY with:
        - 'search' for information requests
        - 'add_numbers' ONLY if numbers are provided
        - 'none' for other requests
        Task: "{prompt}"
        Classification: """
        
        try:
            # Get and clean LLM response
            response = self.llm(llm_prompt)[0]["generated_text"].strip()
            print(f"LLM Raw Output: '{response}'")  # Debug
            tool_name = response.split()[0].lower()
            
            if tool_name in self.tools:
                print(f"LLM selected tool: {tool_name}")
                return tool_name
        except Exception as error:
            print(f"LLM error: {error}")
        
        return ""

    def _try_keyword_fallback(self, prompt: str) -> str:
        """Fallback to keyword matching if LLM selection fails"""
        print("Trying keyword fallback...")
        
        search_pattern = r"\b(search|find|look|query)\b"
        add_pattern = r"\b(add|sum|\+|plus|total)\b"
        
        if re.search(search_pattern, prompt, re.IGNORECASE):
            print("Matched search keywords")
            return "search"
        elif re.search(add_pattern, prompt, re.IGNORECASE):
            print("Matched add keywords")
            return "add_numbers"
        
        print("No matching keywords found")
        return ""

# ===== Main Execution =====
if __name__ == "__main__":
    agent = Agent()
    
    # Test cases
    print(agent.run("Search for cats"))       # Should use search
    print(agent.run("Add 5 and 3"))           # Should use add_numbers
    print(agent.run("What's 2+2+2?"))         # Should use add_numbers via keywords
    print(agent.run("Total 10, 20 and 30"))   # Should use add_numbers via 'total' keyword
    print(agent.run("Tell a joke"))           # Should fail
    print(agent.run("Sum 100 1000 900 0"))    # Passed by LLM
    