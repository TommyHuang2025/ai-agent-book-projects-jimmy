#!/usr/bin/env python3
"""
Main entry point for User Memory System with Separated Architecture
Conversational agent handles dialogue, background processor handles memory
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Optional
from conversational_agent import ConversationalAgent, ConversationConfig
from background_memory_processor import BackgroundMemoryProcessor, MemoryProcessorConfig
from config import Config, MemoryMode
from memory_manager import ensure_memory_cleared

# Add evaluation framework support
# We load it dynamically only when needed to avoid import conflicts
EVALUATION_AVAILABLE = False
UserMemoryEvaluationFramework = None
TestCase = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_result(result: dict):
    """Print formatted result"""
    if result.get('success'):
        print("\n✅ Task completed successfully!")
        if result.get('final_answer'):
            print("\n📝 Final Answer:")
            print("-"*40)
            print(result['final_answer'])
    else:
        print("\n❌ Task failed!")
        if result.get('error'):
            print(f"Error: {result['error']}")
    
    print(f"\n📊 Statistics:")
    print(f"  - Iterations: {result.get('iterations', 0)}")
    print(f"  - Tool calls: {len(result.get('tool_calls', []))}")
    
    if result.get('trajectory_file'):
        print(f"\n💾 Trajectory saved to: {result['trajectory_file']}")
    
    # Show tool call summary
    if result.get('tool_calls'):
        print(f"\n🔧 Tool Call Summary:")
        tool_summary = {}
        for call in result['tool_calls']:
            tool_name = call.tool_name
            if tool_name not in tool_summary:
                tool_summary[tool_name] = {
                    'count': 0,
                    'success': 0,
                    'failed': 0
                }
            tool_summary[tool_name]['count'] += 1
            if call.error:
                tool_summary[tool_name]['failed'] += 1
            else:
                tool_summary[tool_name]['success'] += 1
        
        for tool_name, stats in tool_summary.items():
            print(f"  - {tool_name}: {stats['count']} calls "
                  f"({stats['success']} success, {stats['failed']} failed)")
    
    # Show memory state
    if result.get('memory_state'):
        print(f"\n💭 Memory State:")
        print("-"*40)
        memory_preview = result['memory_state'][:500]
        if len(result['memory_state']) > 500:
            memory_preview += "..."
        print(memory_preview)


def interactive_mode(user_id: str, memory_mode: MemoryMode = MemoryMode.NOTES, 
                    enable_background_processing: bool = True,
                    conversation_interval: int = 1,
                    provider: Optional[str] = None,
                    model: Optional[str] = None,
                    history_limit: int = 10):
    """Run the agent in interactive mode with separated architecture"""
    print_section(f"Interactive Mode - Conversational Agent (User: {user_id})")
    
    # Determine provider and get API key
    provider = (provider or Config.PROVIDER).lower()
    api_key = Config.get_api_key(provider)
    if not api_key:
        print(f"❌ Error: Please set API key for provider '{provider}'")
        # Add provider-specific instructions
        if provider in ["kimi", "moonshot"]:
            print("   export MOONSHOT_API_KEY='your-api-key-here'")
        elif provider == "siliconflow":
            print("   export SILICONFLOW_API_KEY='your-api-key-here'")
        elif provider == "doubao":
            print("   export DOUBAO_API_KEY='your-api-key-here'")
        elif provider == "openrouter":
            print("   export OPENROUTER_API_KEY='your-api-key-here'")
        return
    
    # Initialize conversational agent
    conv_config = ConversationConfig(
        enable_memory_context=True,
        enable_conversation_history=True,
        recent_history_limit=history_limit
    )
    
    agent = ConversationalAgent(
        user_id=user_id,
        api_key=api_key,
        provider=provider,
        model=model,
        config=conv_config,
        memory_mode=memory_mode,
        verbose=True
    )
    
    # Initialize and start background memory processor if enabled
    memory_processor = None
    if enable_background_processing:
        proc_config = MemoryProcessorConfig(
            conversation_interval=conversation_interval,
            min_conversation_turns=1,
            context_window=10,
            enable_auto_processing=True,
            output_operations=True
        )
        
        memory_processor = BackgroundMemoryProcessor(
            user_id=user_id,
            api_key=api_key,
            provider=provider,
            model=model,
            config=proc_config,
            memory_mode=memory_mode,
            verbose=True
        )
        
        memory_processor.start_background_processing()
        print(f"\n🧠 Background memory processing enabled (every {conversation_interval} conversation{'s' if conversation_interval > 1 else ''})")
    
    print("\n✅ Conversational agent initialized")
    print(f"📦 Memory Mode: {memory_mode.value}")
    print(f"🆔 Session: {agent.get_session_id()}")
    print(f"🔄 Background Processing: {'Enabled' if enable_background_processing else 'Disabled'}")
    if enable_background_processing:
        print(f"📊 Processing Trigger: Every {conversation_interval} conversation{'s' if conversation_interval > 1 else ''}")
    print("\nAvailable commands:")
    print("  'memory'  - Show current memory state")
    print("  'process' - Manually trigger memory processing")
    print("  'save'    - Save memory immediately")
    print("  'reset'   - Start new conversation session")
    print("  'quit'    - Exit immediately without saving")
    print("  'exit'    - Exit immediately without saving")
    print("\nOr enter any message to chat.")
    
    conversation_count = 0
    
    while True:
        try:
            print("\n" + "-"*60)
            user_input = input("You > ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                if memory_processor:
                    memory_processor.stop_background_processing()
                print("👋 Goodbye! (Exited without saving)")
                break
            
            elif user_input.lower() == 'save':
                print("\n💾 Saving memory...")
                if memory_processor:
                    results = memory_processor.process_recent_conversations()
                    print(f"✅ Memory saved: {results}")
                else:
                    print("⚠️ Background processing is disabled. Memory is saved after each conversation.")
                continue
            
            elif user_input.lower() == 'memory':
                print("\n💭 Current Memory State:")
                print("-"*40)
                # Explicitly load memory before showing it to the user
                agent.memory_manager.load_memory()
                print(agent.memory_manager.get_context_string())
                
            elif user_input.lower() == 'process':
                if memory_processor:
                    print("\n🔄 Manually triggering memory processing...")
                    results = memory_processor.process_recent_conversations()
                    
                    operations = results.get('operations', [])
                    if operations:
                        print(f"\n📝 Memory Operations ({len(operations)} total):")
                        for i, op in enumerate(operations, 1):
                            icon = {'add': '➕', 'update': '📝', 'delete': '🗑️'}.get(op['action'], '❓')
                            print(f"{i}. {icon} {op['action'].upper()}: {op.get('content', op.get('memory_id', 'N/A'))}")
                    else:
                        print("ℹ️ No memory updates needed")
                    
                    summary = results.get('summary', {})
                    print(f"\nSummary: {summary.get('added', 0)} added, {summary.get('updated', 0)} updated, {summary.get('deleted', 0)} deleted")
                else:
                    print("❌ Background processing not enabled")
                    
            elif user_input.lower() == 'reset':
                agent.reset_session()
                print("✅ Started new conversation session")
                conversation_count = 0
                
            else:
                # Explicitly load memories before chatting
                agent.memory_manager.load_memory()
                response = agent.chat(user_input)
                conversation_count += 1
                
                if memory_processor:
                    memory_processor.increment_conversation_count()
                    
                    if memory_processor.should_process():
                        print(f"\n[Memory processing triggered after {conversation_interval} conversation{'s' if conversation_interval > 1 else ''}]")
                        time.sleep(2)
                    elif conversation_interval > 1:
                        conversations_until_process = conversation_interval - (conversation_count % conversation_interval)
                        if conversations_until_process < conversation_interval:
                            print(f"\n[Memory processing in {conversations_until_process} more conversation{'s' if conversations_until_process > 1 else ''}]")
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted. Type 'save' to save memory, or 'quit'/'exit' to exit immediately without saving.")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            logger.error(f"Error in interactive mode: {e}", exc_info=True)
    
    if memory_processor:
        memory_processor.stop_background_processing()


def demo_memory_system(memory_mode: MemoryMode = None, 
                       provider: Optional[str] = None, 
                       model: Optional[str] = None,
                       history_limit: int = 10):
    """Demonstrate the separated memory system architecture"""
    print_section("Demo: Separated Memory Architecture")
    
    # Determine provider and get API key
    provider = (provider or Config.PROVIDER).lower()
    api_key = Config.get_api_key(provider)
    if not api_key:
        print(f"❌ Please set API key for provider '{provider}'")
        return
    
    # Create test user
    user_id = "demo_user"
    
    # Use provided memory_mode or prompt for it
    if memory_mode is None:
        memory_mode = select_memory_mode_interactive()
    
    # Initialize conversational agent
    conv_config = ConversationConfig(
        enable_memory_context=True,
        enable_conversation_history=True,
        recent_history_limit=history_limit
    )
    
    agent = ConversationalAgent(
        user_id=user_id,
        api_key=api_key,
        provider=provider,
        model=model,
        config=conv_config,
        memory_mode=memory_mode,
        verbose=True
    )
    
    # Initialize background processor
    proc_config = MemoryProcessorConfig(
        conversation_interval=2,  # Process every 2 conversations for demo
        min_conversation_turns=1,
        output_operations=True
    )
    
    processor = BackgroundMemoryProcessor(
        user_id=user_id,
        api_key=api_key,
        provider=provider,
        model=model,
        config=proc_config,
        memory_mode=memory_mode,
        verbose=True
    )
    
    # CRITICAL: Clear memory and history before starting demo
    print("\n🧹 Clearing previous demo memories and history...")
    if hasattr(agent.memory_manager, 'clear_all_memories'):
        agent.memory_manager.clear_all_memories()
    if hasattr(processor.memory_manager, 'clear_all_memories'):
        processor.memory_manager.clear_all_memories()
    
    # Also clear conversation history for a clean run
    if agent.conversation_history:
        agent.conversation_history.conversations = []
        agent.conversation_history.save_history()
    
    print("✅ Memories and history cleared.")

    # Session 1: Have conversations
    print("\n📝 Session 1: Having conversations")
    print("-"*40)
    
    messages = [
        "Hi! My name is Alice and I work as a product manager at TechCorp.",
        "I prefer Python for scripting and use VS Code as my IDE. I also like dark themes.",
        "I'm currently working on a new mobile app project for our company."
    ]
    
    for message in messages:
        print(f"\n👤 User: {message}")
        # In demo, we don't need to print the live response, just get it
        _ = agent.chat(message)
        print(f"🤖 Assistant: (Responded to user)")
        time.sleep(1)
    
    # Process memories
    print("\n\n🔄 Processing conversation for memory updates...")
    print("-"*40)

    for _ in range(len(messages)):
        processor.increment_conversation_count()

    results = processor.process_recent_conversations()
    
    # Display operations from the background processor's analysis agent
    tool_calls = results.get('tool_calls', [])
    if tool_calls:
        print(f"\n🛠️  Memory Operations Detected ({len(tool_calls)} total):")
        for i, call in enumerate(tool_calls, 1):
            icon = {'add_memory': '➕', 'update_memory': '📝', 'delete_memory': '🗑️'}.get(call.tool_name, '❓')
            content = call.arguments.get('content', call.arguments.get('memory_id', 'N/A'))
            print(f"{i}. {icon} {call.tool_name}: {content}")
    else:
        print("ℹ️ No memory updates were needed based on the conversation.")
    
    summary = results.get('summary', {})
    print(f"\n✅ Summary: {summary.get('added', 0)} added, {summary.get('updated', 0)} updated, {summary.get('deleted', 0)} deleted")
    # Start new session to test memory persistence
    print("\n\n📝 Session 2: Testing memory persistence in a new session")
    print("-"*40)
    
    agent.reset_session()
    
    # Explicitly load memories from processor before testing
    print("\n🔄 Agent is loading memories for the new session...")
    agent.memory_manager.load_memory()
    print("✅ Memories loaded.")
    
    test_message = "What do you know about me and my work?"
    print(f"\n👤 User: {test_message}")
    
    # Turn on verbose for the agent to see the context being used
    agent.verbose = True
    response = agent.chat(test_message)
    agent.verbose = False # Turn it back off
    
    print("\n🤖 Assistant Response:")
    print(response)
    
    # Show final memory state
    print("\n\n💭 Final Memory State:")
    print("-"*40)
    print(agent.memory_manager.get_context_string())


def run_evaluation_mode(user_id: str, memory_mode: MemoryMode, verbose: bool = True, 
                        provider: Optional[str] = None, model: Optional[str] = None,
                        history_limit: int = 10):
    """Run evaluation mode using the evaluation framework"""
    
    # Import the evaluation framework with proper module isolation
    from pathlib import Path
    
    eval_framework_path = Path(__file__).parent.parent / "user-memory-evaluation"
    
    try:
        # Save the current modules to avoid conflicts
        saved_modules = {}
        conflicting_modules = ['config', 'models', 'evaluator', 'framework']
        
        # Temporarily remove conflicting modules from sys.modules
        for module_name in conflicting_modules:
            if module_name in sys.modules:
                saved_modules[module_name] = sys.modules[module_name]
                del sys.modules[module_name]
        
        # Temporarily add evaluation framework path with highest priority
        original_path = sys.path.copy()
        sys.path.insert(0, str(eval_framework_path))
        
        # Import evaluation framework modules
        import config as eval_config
        import models as eval_models  
        import evaluator as eval_evaluator
        import framework as eval_framework
        
        # Get the class we need
        framework_class = eval_framework.UserMemoryEvaluationFramework
        
        # Restore original path
        sys.path = original_path
        
        # Remove evaluation modules from sys.modules to avoid future conflicts
        for module_name in conflicting_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        # Restore original modules
        for module_name, module in saved_modules.items():
            sys.modules[module_name] = module
            
    except Exception as e:
        # Restore on error
        sys.path = original_path if 'original_path' in locals() else sys.path
        for module_name, module in saved_modules.items():
            sys.modules[module_name] = module
            
        print(f"❌ Error: Could not load evaluation framework: {e}")
        print("Please ensure user-memory-evaluation is properly installed.")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print_section("Evaluation Mode - Test Case Based Evaluation")
    
    # Initialize evaluation framework
    framework = framework_class()
    
    if not framework.test_suite:
        print("❌ Error: No test cases loaded")
        sys.exit(1)
    
    print(f"\n✅ Loaded {len(framework.test_suite.test_cases)} test cases")
    
    provider = (provider or Config.PROVIDER).lower()
    api_key = Config.get_api_key(provider)
    if not api_key:
        print(f"❌ Error: Please set API key for provider '{provider}'")
        sys.exit(1)
    
    # Initialize agents for the evaluation
    conv_config = ConversationConfig(recent_history_limit=history_limit)
    agent = ConversationalAgent(
        user_id=user_id, api_key=api_key, provider=provider, model=model,
        config=conv_config, memory_mode=memory_mode, verbose=verbose
    )
    processor = BackgroundMemoryProcessor(
        user_id=user_id, api_key=api_key, provider=provider, model=model,
        config=MemoryProcessorConfig(), memory_mode=memory_mode, verbose=verbose
    )
    
    while True:
        print("\n" + "-"*60)
        print("Options:")
        print("1. Run a test case")
        print("2. View current memory state")
        print("3. View current conversations")
        print("4. Clear all memory")
        print("5. Clear all conversations")
        print("6. Exit evaluation mode")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            print("\n📋 Available Test Cases:")
            framework.display_test_case_summary(show_full_titles=True, by_category=True)
            
            test_id = input("\nEnter test case ID to run (or 'cancel' to go back): ").strip()
            if test_id.lower() == 'cancel':
                continue
            
            test_case = framework.get_test_case(test_id)
            if not test_case:
                print(f"❌ Test case '{test_id}' not found")
                continue
            
            print(f"\n{'='*60}")
            print(f"Running Test Case: {test_case.title}")
            print(f"Category: {test_case.category}")
            print("="*60)
            
            print("\n🧹 Clearing all memory and state before test...")
            ensure_memory_cleared(agent.memory_manager, "agent memory")
            ensure_memory_cleared(processor.memory_manager, "processor memory")
            if agent.conversation_history:
                agent.conversation_history.conversations = []
                agent.conversation_history.save_history()
            agent.reset_session()
            print("✅ All state cleared.")

            print(f"\n📚 Simulating {len(test_case.conversation_histories)} past conversations...")
            conversation_contexts = []
            for history in test_case.conversation_histories:
                context = [{"role": msg.role.value, "content": msg.content} for msg in history.messages]
                conversation_contexts.append(context)

            if conversation_contexts:
                print(f"\n🧠 Processing conversations to build memory...")
                results = processor.process_conversation_batch(conversation_contexts)
                total_added = sum(r.get('summary', {}).get('added', 0) for r in results)
                print(f"  ✅ Memory processing complete. Added {total_added} memories.")
            
            print("\n🔄 Starting a new, fresh session for the test question.")
            agent.reset_session()
            agent.memory_manager.load_memory() # Reload memory saved by processor
            
            memory_context = agent.memory_manager.get_context_string()
            print("\n💾 Available memories for this session:")
            print("-"*40)
            print(memory_context[:500] + "..." if len(memory_context) > 500 else memory_context)
            print("-"*40)
            
            print(f"\n{'='*60}")
            print("USER QUESTION:")
            print(test_case.user_question)
            print("="*60)
            
            print("\n🤔 Generating response based ONLY on structured memories...")
            response = agent.chat(test_case.user_question)
            
            print("\n📝 Agent Response:")
            print("-"*60)
            print(response)
            print("-"*60)
            
            print("\n⚖️ Evaluating response...")
            eval_result = framework.submit_and_evaluate(test_id, response)
            
            if eval_result:
                status = "✅ PASSED" if eval_result.passed else "❌ FAILED"
                print(f"\n{'='*60}\nEVALUATION RESULT:\n{'-'*60}")
                print(f"Status: {status}")
                print(f"Reward Score: {eval_result.reward:.3f}/1.000")
                if eval_result.reasoning: print(f"\nReasoning:\n{eval_result.reasoning}")
                if eval_result.suggestions: print(f"\nSuggestions:\n{eval_result.suggestions}")
                print("="*60)
            else:
                print("❌ Evaluation failed to run.")

        elif choice == "2":
            print("\n📄 Current Memory State:")
            print("-"*60)
            print(agent.memory_manager.get_context_string())
        
        elif choice == "3":
            print("\n💬 Current Conversation History:")
            print("-"*60)
            if agent.conversation_history:
                agent.conversation_history.load_history()
                if agent.conversation_history.conversations:
                    for turn in agent.conversation_history.conversations:
                        print(f"[{turn.timestamp} - Session: {turn.session_id}]")
                        print(f"  You > {turn.user_message}")
                        print(f"  Agent > {turn.assistant_message}\n")
                else:
                    print("No conversation history found.")
            else:
                print("Conversation history is not enabled.")
                
        elif choice == "4":
            if input("\n⚠️ Are you sure you want to clear all memory? (yes/no): ").lower() == "yes":
                ensure_memory_cleared(agent.memory_manager, "agent memory")
                ensure_memory_cleared(processor.memory_manager, "processor memory")
                print("✅ Memory cleared.")

        elif choice == "5":
            if input("\n⚠️ Are you sure you want to clear all conversation history? (yes/no): ").lower() == "yes":
                if agent.conversation_history:
                    agent.conversation_history.conversations = []
                    agent.conversation_history.save_history()
                    print("✅ Conversation history cleared.")
                else:
                    print("ℹ️ Conversation history is not enabled.")
                
        elif choice == "6":
            print("\nExiting evaluation mode...")
            break
        else:
            print(f"❌ Invalid choice: {choice}")


def select_mode_interactive() -> str:
    """
    Interactively prompt the user to select an execution mode
    
    Returns:
        Selected mode string ('evaluation', 'interactive', or 'demo')
    """
    print("\n" + "="*60)
    print("  🚀 SELECT EXECUTION MODE")
    print("="*60)
    
    print("\n1. Evaluation Mode")
    print("   - Run test cases from user-memory-evaluation framework")
    print("   - Test memory system with predefined scenarios")
    print("   - Get performance scores and feedback")
    
    print("\n2. Interactive Mode")
    print("   - Chat with the agent in real-time")
    print("   - Memory processes automatically in background")
    print("   - Commands: memory, process, save, reset, quit/exit")
    
    print("\n3. Demo Mode")
    print("   - Quick demonstration of memory system")
    print("   - Shows how conversations are processed into memories")
    print("   - Tests memory persistence across sessions")
    
    print("\n" + "-"*60)
    
    while True:
        try:
            choice = input("\nSelect mode (1-3): ").strip()
            
            if choice == '1':
                print("✅ Selected: Evaluation Mode")
                return "evaluation"
            elif choice == '2':
                print("✅ Selected: Interactive Mode")
                return "interactive"
            elif choice == '3':
                print("✅ Selected: Demo Mode")
                return "demo"
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Operation cancelled by user")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")


def select_memory_mode_interactive() -> MemoryMode:
    """
    Interactively prompt the user to select a memory mode
    
    Returns:
        Selected MemoryMode
    """
    print("\n" + "="*60)
    print("  📝 SELECT MEMORY MODE")
    print("="*60)
    
    print("\n1. Simple Notes (Basic)")
    print("   - Store simple facts and preferences")
    print("   - Each memory is a single line or fact")
    print("   - Example: 'User email: john@example.com'")
    
    print("\n2. Enhanced Notes")
    print("   - Store comprehensive contextual information")
    print("   - Each memory can be a full paragraph with context")
    print("   - Example: 'User works at TechCorp as a senior engineer,")
    print("             specializing in ML for 3 years...'")
    
    print("\n3. JSON Cards (Basic)")
    print("   - Hierarchical structured memory")
    print("   - Format: category → subcategory → key → value")
    print("   - Example: personal.contact.email → 'john@example.com'")
    
    print("\n4. Advanced JSON Cards")
    print("   - Complete memory card objects with metadata")
    print("   - Each card includes backstory, person, relationship")
    print("   - Prevents confusion between different contexts")
    print("   - Example: Medical card for child vs elderly parent")
    
    print("\n" + "-"*60)
    
    while True:
        try:
            choice = input("\nSelect mode (1-4): ").strip()
            
            if choice == '1':
                print("✅ Selected: Simple Notes Mode")
                return MemoryMode.NOTES
            elif choice == '2':
                print("✅ Selected: Enhanced Notes Mode")
                return MemoryMode.ENHANCED_NOTES
            elif choice == '3':
                print("✅ Selected: JSON Cards Mode")
                return MemoryMode.JSON_CARDS
            elif choice == '4':
                print("✅ Selected: Advanced JSON Cards Mode")
                return MemoryMode.ADVANCED_JSON_CARDS
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Operation cancelled by user")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function with command-line argument support"""
    parser = argparse.ArgumentParser(
        description="User Memory Agent with React Pattern - Following system-hint architecture"
    )
    
    parser.add_argument(
        "--mode",
        choices=["interactive", "demo", "evaluation"],
        default=None,
        help="Execution mode (if not specified, prompts interactively)"
    )
    
    parser.add_argument(
        "--user",
        type=str,
        default="default_user",
        help="User ID for memory system (default: default_user)"
    )
    
    
    parser.add_argument(
        "--background-processing",
        type=bool,
        default=True,
        help="Enable background memory processing (default: True)"
    )
    
    parser.add_argument(
        "--conversation-interval",
        type=int,
        default=1,
        help="Process memory after N conversations (default: 1 - every conversation)"
    )
    
    parser.add_argument(
        "--memory-mode",
        choices=["notes", "enhanced_notes", "json_cards", "advanced_json_cards"],
        help="Memory mode (prompts interactively if not specified)"
    )
    
    parser.add_argument(
        "--provider",
        choices=["siliconflow", "doubao", "kimi", "moonshot", "openrouter"],
        default=None,
        help="LLM provider (defaults to env PROVIDER or 'kimi')"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (defaults to provider's default model)"
    )

    parser.add_argument(
        "--history-limit", 
        type=int, 
        default=10, 
        help="Number of recent conversation turns to include in context"
    )
    
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable verbose output (verbose is enabled by default)"
    )
    
    args = parser.parse_args()
    
    # Set verbose based on no-verbose flag (default is verbose=True)
    verbose = not args.no_verbose
    
    # Determine provider
    provider = args.provider or Config.PROVIDER
    
    # Validate configuration
    if not Config.validate(provider):
        sys.exit(1)
    
    # Create necessary directories
    Config.create_directories()
    
    # Select execution mode if not specified
    execution_mode = args.mode
    if execution_mode is None:
        # Prompt user to select mode
        execution_mode = select_mode_interactive()
    
    # Configure memory mode
    if args.memory_mode:
        # Mode specified via command line
        mode_map = {
            "notes": MemoryMode.NOTES,
            "enhanced_notes": MemoryMode.ENHANCED_NOTES,
            "json_cards": MemoryMode.JSON_CARDS,
            "advanced_json_cards": MemoryMode.ADVANCED_JSON_CARDS
        }
        memory_mode = mode_map[args.memory_mode]
    else:
        # Interactive mode selection
        memory_mode = select_memory_mode_interactive()
    
    print("\n" + "🧠"*40)
    print("  USER MEMORY SYSTEM - SEPARATED ARCHITECTURE")
    print("🧠"*40)
    
    if execution_mode == "demo":
        demo_memory_system(memory_mode, provider, args.model, history_limit=args.history_limit)
    
    elif execution_mode == "evaluation":
        run_evaluation_mode(args.user, memory_mode, verbose, provider, args.model, history_limit=args.history_limit)
    
    elif execution_mode == "interactive":
        interactive_mode(
            user_id=args.user,
            memory_mode=memory_mode,
            enable_background_processing=args.background_processing,
            conversation_interval=args.conversation_interval,
            provider=provider,
            model=args.model,
            history_limit=args.history_limit
        )
    
    else:
        # This should not happen, but handle it gracefully
        print(f"❌ Unknown execution mode: {execution_mode}")
        sys.exit(1)
    
    print("\n👋 Thank you for using User Memory Agent!")


if __name__ == "__main__":
    main()