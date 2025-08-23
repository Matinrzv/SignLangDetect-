import pygame
import numpy as np

class SignLanguageApp:
    def __init__(self, width, height):
        pygame.init()
        self.font = pygame.font.SysFont('Arial', 32)
        self.text_color = (0, 0, 0)
        self.bg_color = (255, 255, 255)
        self.window = pygame.display.set_mode((width, height + 50))
        pygame.display.set_caption("Sign Language Recognition")
        self.predicted_text = ''
        self.height = height

    def update_frame(self, frame, prediction_char=None, bbox=None):
        if prediction_char:
            if len(self.predicted_text) == 0 or prediction_char != self.predicted_text[-1]:
                self.predicted_text += prediction_char
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

        self.window.blit(frame_surface, (0, 0))

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            rect_color = (255, 0, 0)  
            rect_thickness = 3
            pygame.draw.rect(self.window, rect_color, pygame.Rect(x1, y1, x2 - x1, y2 - y1), rect_thickness)

        pygame.draw.rect(self.window, self.bg_color, (0, self.height, frame.shape[1], 50))
        text_surface = self.font.render(self.predicted_text, True, self.text_color)
        self.window.blit(text_surface, (10, self.height + 10))
        pygame.display.update()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e:
                    return False
                elif event.key == pygame.K_c:
                    self.predicted_text = ''
        return True

    def quit(self):
        pygame.quit()
