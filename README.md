# Agentic AI System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent agent that routes user requests using hybrid semantic+LLM+keyword decision making.

## Key Features
- **Three-Stage Routing**:
  - Semantic matching with `all-MiniLM-L6-v2`
  - LLM classification (`google/flan-t5-large`)
  - Regex keyword fallback
- **Decorator-based Tools**:
  ```python
  @agent._tool
  def your_tool(query): 
      """Docstring powers semantic matching"""

  
