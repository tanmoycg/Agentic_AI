from transformers import pipeline
from typing import Dict, Callable
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer  # New import for context


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
    def __init__(self, encoder=None):
        self.llm = pipeline(**LLM_SETTINGS)
        self.tools = self._initialize_tools()
        self.memory = defaultdict(list)  # NEW: User memory storage
        self.encoder = encoder or SentenceTransformer("all-MiniLM-L6-v2")
        self._setup_tool_embeddings()    # NEW: Precompute tool embeddings

     # NEW: Initialize semantic tool matching
    def _setup_tool_embeddings(self):
        """Create embeddings for all tool descriptions"""
        self.tool_embeddings = {
            name: self.encoder.encode(func.__doc__)
            for name, func in self.tools.items()
        }

    def run(self, user_id: str, prompt: str) -> str:  # MODIFIED: Added user_id
        """
        Process input with memory and context.
        Changes:
        - Now tracks user-specific history
        - Augments prompt with contextual embeddings
        """
        # NEW: Augment prompt with memory and semantic context
        context = (
            f"Last tools used: {self.memory[user_id][-2:]}\n"
            f"Current query: {prompt}"
        )
        tool_name = self._choose_tool_with_context(prompt, context)  # MODIFIED
        result = self.tools[tool_name](prompt)
        
        # NEW: Update memory
        self.memory[user_id].append((prompt, tool_name))
        return result


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
                Mathematical addition for phrases containing: 
                - Explicit commands: 'add', 'sum', 'total', 'plus', '+'
                - Equations: '2+3', '5 and 6'
                - Implicit math: 'combine these numbers'
                Example inputs: 'Add 5 and 3', 'Total 10 and 20', '2+2'
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

     # NEW: Context-aware tool selection
    def _choose_tool_with_context(self, prompt: str, context: str) -> str:
        """
        Hybrid selection using:
        1. Semantic matching (context)
        2. LLM classification (prompt)
        3. Keyword fallback
        """
        # Try semantic match first
        prompt_embed = self.encoder.encode(prompt)
        similarities = {
            name: prompt_embed.dot(tool_embed)  # Cosine similarity
            for name, tool_embed in self.tool_embeddings.items()
        }
        semantic_tool = max(similarities, key=similarities.get)
        
        if similarities[semantic_tool] > 0.3:  # Threshold
            print(f"Semantic match: {semantic_tool}")
            return semantic_tool
        
        # Fallback to original LLM + keyword flow
        return self._choose_tool(f"Context:\n{context}\nQuery:{prompt}")


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
    
    # Test with user context
    user = "test_user"
    print(agent.run(user, "Search for cats"))       # Semantic → search
    print(agent.run(user, "Add 5 and 3"))           # Semantic → add_numbers
    print(agent.run(user, "Find dog breeds"))       # Memory-augmented
    print(agent.run(user, "Total 10 and 20"))       # Uses previous 'add' context
    print(agent.run(user, "combine 10000 and 20"))
    print(agent.run(user, "What's the sum of the first 10 prime numbers multiplied by 3?"))
    print(agent.run(user, "Where is Kolakta"))