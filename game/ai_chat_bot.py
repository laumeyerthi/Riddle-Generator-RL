import threading
import queue
from google import genai
from llm_interface.ppo_interface import PPOInterface
from llm_interface.ppo_masked_interface import PPOMaskedInterface
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY=os.getenv("GEMINI_API_KEY")

class AIChatBot:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = True
        self.latest_game_state = {}
        self.current_mask = None
        self.interface = PPOInterface()
        self.interface_mask = PPOMaskedInterface()
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY not found! Check your .env file.")
        self.client = genai.Client(api_key=API_KEY)
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
    
    
    def send_message(self, user_text, game_state, mask = None):
        self.latest_game_state = game_state
        self.input_queue.put(user_text)
        if(mask is not None):
            self.current_mask = mask
        
    def get_new_messages(self):
        messages = []
        while not self.output_queue.empty():
            messages.append(self.output_queue.get())
        return messages
    
    def _worker_loop(self):
        
        try:
            chat = self.client.chats.create(model="gemini-flash-latest")        
        except Exception as e:
            print(f"Failed to create chat: {e}")
            return
        while self.running:
            try:
                try:
                    user_text = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                context_str = ""
                if self.current_mask is None:
                    context_str = f"""
                        [SYSTEM CONTEXT]
                        Current Game State: {self.latest_game_state}
                        Current Action Prediction : {self.interface.get_action(self.latest_game_state)}
                        Current Action Propabilities: {self.interface.get_action_probs(self.latest_game_state)}
                        Current Advantages : {self.interface.get_winning_probs(self.latest_game_state)}
                        User Input: {user_text}
                        
                        Instruction: You are a helpful game copilot to a labirynth game. You have the results from an ppo rl agent trained on the game. Keep advice short (under 2 sentences). The actions are in order right, up, left, down, backtrack, button1, button2, button3 and button4.
                        """
                else:
                    context_str = f"""
                        [SYSTEM CONTEXT]
                        Current Game State: {self.latest_game_state}
                        Current Allowed Actions: {self.current_mask}
                        Current Action Prediction : {self.interface_mask.get_action(self.latest_game_state, self.current_mask)}
                        Current Action Propabilities: {self.interface.get_action_probs(self.latest_game_state)}
                        Current Advantages : {self.interface.get_winning_probs(self.latest_game_state)}
                        User Input: {user_text}
                        
                        Instruction: You are a helpful game copilot to a labirynth game. You have the results from an ppo rl agent trained on the game. Keep advice short (under 2 sentences). The actions are in order right, up, left, down, backtrack, button1, button2, button3 and button4.
                        """
                                
                response = chat.send_message(context_str)
                   
                ai_text = response.text
                if ai_text:
                    clean_text = ai_text.strip().replace("**", "")
                    self.output_queue.put(f"Gemini: {clean_text}")
            except Exception as e:
                print(f"API ERROR: {e}")
                self.output_queue.put("System: AI Error (See Console)")