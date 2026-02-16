from gymnasium.utils.play import play
import pygame
import sys
import os
from ai_chat_bot import AIChatBot
import numpy as np

# Add parent directory for env import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gymnasium_env.envs.lab_env import LabEnv

mapping = {
    pygame.K_RIGHT:0,
    pygame.K_UP:1,
    pygame.K_LEFT:2,
    pygame.K_DOWN:3,
    pygame.K_LALT:4,
    pygame.K_1:5,
    pygame.K_2:6,
    pygame.K_3:7,
    pygame.K_4:8,
}

def wrap_text(text, font, max_width):
    words = text.split(' ')
    lines = []
    current_line = []

    for word in words:
        # Test what the line would look like with the new word
        test_line = ' '.join(current_line + [word])
        width, _ = font.size(test_line)
        
        if width <= max_width:
            current_line.append(word)
        else:
            # Line is too long, save the current one and start a new one
            lines.append(' '.join(current_line))
            current_line = [word]
            
    # Add the last line
    if current_line:
        lines.append(' '.join(current_line))
    return lines

def play_game():
    env = LabEnv(render_mode="rgb_array")
    ai_chat = AIChatBot()
    observation, info = env.reset()
    
    pygame.init()
    frame = env.render() 
    screen_width, screen_height = frame.shape[1], frame.shape[0]
    screen = pygame.display.set_mode((screen_width, screen_height+400))
    
    # screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Labyrinth")
    font = pygame.font.Font(None, 24)
    pygame.display.set_caption("Labyrinth")
    
    
    
    user_text = ""
    chat_history = ["System: Ask me for tips!"]
    typing_mode = False    
    running = True
    current_state = observation
    
    while running:
        event = pygame.event.wait(timeout=100)
        
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if typing_mode:
                    print(user_text)
                    ai_chat.send_message(user_text, current_state, env.action_masks())
                    user_text = ""
                    typing_mode = False
                else:
                    typing_mode = True
            elif typing_mode:
                if event.key == pygame.K_BACKSPACE:
                    user_text = user_text [:-1]
                else:
                    user_text += event.unicode
            else:
                if event.key in mapping:
                    current_action = mapping[event.key]
            
                    current_state, reward, terminated, truncated, info = env.step(current_action)

                    if(terminated or truncated):
                        print("Game Over! Resetting...")
                        current_state, info = env.reset()
        new_msg = ai_chat.get_new_messages()
        if new_msg:
            chat_history.extend(new_msg)
            print("New Message")
            
        frame = env.render()
        frame = np.swapaxes(frame, 0,1)
        frame_surf = pygame.surfarray.make_surface(frame)
        screen.blit(frame_surf,(0,0))
        
        pygame.draw.rect(screen, (0, 0, 0), (0, screen_height, screen_width, 400))
        y_offset = screen_height+10
        for msg in chat_history[-5:]:
            wrapped_lines = wrap_text(msg, font, screen_width-20)
    
            for line in wrapped_lines:
                text_surf = font.render(line, True, (255, 255, 255))
                screen.blit(text_surf, (10, y_offset))
                
                y_offset += 25
                
                if y_offset > (screen_height + 350):
                    break
        
        if typing_mode:
            display_text = f"Chat: {user_text}_" 
            color = (0, 255, 0) 
        else:
            display_text = "Press ENTER to chat"
            color = (150, 150, 150) 
        input_surf = font.render(display_text, True, (0, 255, 0))
        screen.blit(input_surf, (10, screen_height+360))

        pygame.display.flip()
        
        
    env.close()
    pygame.quit()
    

    
if __name__ == "__main__":
    play_game()