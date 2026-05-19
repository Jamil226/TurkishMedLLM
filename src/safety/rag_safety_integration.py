#!/usr/bin/env python3
"""
RAG + Safety Prompts Integration Example
Shows how to use SafetyPrompts with TurkishMedRAG for Configuration 5
"""

import sys
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.rag.rag import TurkishMedRAG
from src.safety.safety_prompts import SafetyPrompts, SafetyConfig, SafetyLevel, get_safety_config


class SafeRAG:
    """
    Turkish Medical RAG with Safety Prompts Integration
    Combines RAG responses with comprehensive safety disclaimers
    """
    
    def __init__(self, safety_level: str = "high"):
        """
        Initialize SafeRAG system
        
        Args:
            safety_level: "low", "medium", or "high"
        """
        self.rag = TurkishMedRAG()
        self.safety_config = get_safety_config(safety_level)
        self.safety_prompts = SafetyPrompts(self.safety_config)
        self.safety_level = safety_level
    
    def answer(self, query: str) -> Dict[str, any]:
        """
        Generate RAG answer with safety prompts
        
        Args:
            query: Medical question in Turkish
        
        Returns:
            Dictionary with:
            - query: Original query
            - rag_response: Response from RAG
            - safe_response: Response with safety prompts
            - safety_level: Applied safety level
            - source: Retrieved document source
        """
        # Get RAG response
        rag_response = self.rag.answer(query)
        
        # Apply safety prompts
        safe_response = self.safety_prompts.process_response(
            rag_response['answer']
        )
        
        return {
            'query': query,
            'rag_response': rag_response['answer'],
            'safe_response': safe_response,
            'safety_level': self.safety_level,
            'source': rag_response.get('source', 'Unknown'),
            'contexts': rag_response.get('contexts', [])
        }


# Example usage configurations

CONFIGURATIONS = {
    "1_base_qwen": {
        "description": "Base Qwen3:8B (No RAG, No Safety)",
        "use_rag": False,
        "use_safety": False,
    },
    "2_base_qwen_rag": {
        "description": "Base Qwen3:8B + RAG (No Safety)",
        "use_rag": True,
        "use_safety": False,
    },
    "3_finetuned_qwen": {
        "description": "Fine-tuned Qwen3:8B (No RAG, No Safety)",
        "use_rag": False,
        "use_safety": False,
        "checkpoint": "qlora_checkpoints/final_model"
    },
    "4_finetuned_qwen_rag": {
        "description": "Fine-tuned Qwen3:8B + RAG (No Safety)",
        "use_rag": True,
        "use_safety": False,
        "checkpoint": "qlora_checkpoints/final_model"
    },
    "5_finetuned_qwen_rag_safety": {
        "description": "Fine-tuned Qwen3:8B + RAG + Safety Prompts (HIGH)",
        "use_rag": True,
        "use_safety": True,
        "safety_level": "high",
        "checkpoint": "qlora_checkpoints/final_model"
    }
}


def test_safety_integration():
    """Test the SafeRAG integration"""
    
    print("=" * 80)
    print("🔬 SafeRAG Integration Test")
    print("=" * 80)
    
    # Test query
    test_query = "Diyabet hastalarında kan şekeri neden yükselir?"
    
    # Create SafeRAG instance
    print("\n📌 Initializing SafeRAG (HIGH safety level)...")
    safe_rag = SafeRAG(safety_level="high")
    
    print("✅ SafeRAG initialized successfully")
    print(f"📋 Test Query: {test_query}")
    print("\n" + "-" * 80)
    
    # Get response
    print("\n⏳ Generating response with safety prompts...")
    result = safe_rag.answer(test_query)
    
    print("\n📊 RESULT:")
    print(f"Safety Level: {result['safety_level']}")
    print(f"Source: {result['source']}")
    print("\n" + "=" * 80)
    print("RESPONSE WITH SAFETY PROMPTS:")
    print("=" * 80)
    print(result['safe_response'])
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Import Dict for type hints
    from typing import Dict
    
    print("\n" + "=" * 80)
    print("🛡️  TURKISH MEDICAL LLM - SAFETY PROMPTS INTEGRATION")
    print("=" * 80)
    
    print("\n📋 AVAILABLE CONFIGURATIONS:")
    print("-" * 80)
    for config_name, config in CONFIGURATIONS.items():
        print(f"\n{config_name}:")
        print(f"  Description: {config['description']}")
        print(f"  Use RAG: {config['use_rag']}")
        print(f"  Use Safety: {config['use_safety']}")
        if config['use_safety']:
            print(f"  Safety Level: {config.get('safety_level', 'N/A')}")
        if 'checkpoint' in config:
            print(f"  Checkpoint: {config['checkpoint']}")
    
    print("\n" + "=" * 80)
    print("✅ Safety Prompts System Ready!")
    print("=" * 80)
    print("\nUsage Example:")
    print("  from src.safety.rag_safety_integration import SafeRAG")
    print("  safe_rag = SafeRAG(safety_level='high')")
    print("  result = safe_rag.answer('Your Turkish medical question')")
    print("  print(result['safe_response'])")
    print("\n" + "=" * 80)
