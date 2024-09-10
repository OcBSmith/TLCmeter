from kivy.config import Config
Config.set('kivy', 'log_level', 'info')  # Cambia el nivel de registro a 'info'
Config.set('graphics', 'width', '1280')
Config.set('graphics', 'height', '720')
Config.set('graphics', 'resizable', False)
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

import os
import psutil
import cv2
import numpy as np
from PIL import Image as PilImage
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.spinner import Spinner
from kivy.core.window import Window
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.utils import platform
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivymd.app import MDApp
from kivymd.uix.button import MDRectangleFlatButton
from kivymd.uix.filemanager import MDFileManager
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from pathlib import Path
from kivy.graphics import InstructionGroup
from kivymd.uix.button import *



if platform != 'android':
    import tkinter as tk
    from tkinter import filedialog

# Define el directorio actual y la carpeta 'temp'
current_dir = os.path.dirname(os.path.abspath(__file__))
temp_dir = os.path.join(current_dir, 'temp')

# Verifica si la carpeta 'temp' existe, y si no, la crea
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Ruta de la imagen de ejemplo
image_path = os.path.join(current_dir, 'image.png')

def normalize_path(path):
    return str(Path(path).resolve())

def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype('int').tolist()

def find_dest(corners):
    (tl, tr, br, bl) = corners
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - br[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - tl[1]) ** 2) + ((tl[1] - tl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    return [
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ]

# Función para guardar imágenes en la carpeta 'temp'
def save_image(image, filename):
    filepath = os.path.join(temp_dir, filename)
    cv2.imwrite(filepath, image)

class Magnifier(Widget):
    def __init__(self, img, image_rect, **kwargs):
        super(Magnifier, self).__init__(**kwargs)
        self.img = img
        self.image_rect = image_rect
        self.base_zoom_level = 2.0
        self.base_lens_size = 100
        self.zoom_level = self.base_zoom_level
        self.lens_size = self.base_lens_size
        self.mouse_pos = (0, 0)

        with self.canvas.after:
            Color(1, 1, 1)
            self.lens = Ellipse(size=(self.lens_size, self.lens_size), segments=360)
            Color(0, 0, 0)
            self.lens_border = Line(circle=(0, 0, self.lens_size / 2), width=2, segments=360)
            self.center_point = Ellipse(size=(5, 5))

        self.update_lens_position(0)
        Clock.schedule_interval(self.update_lens_position, 1 / 60)

    def update_lens_position(self, dt):
        pos = self.mouse_pos
        self.lens_border.circle = (pos[0], pos[1], self.lens_size / 2)
        self.lens.pos = (pos[0] - self.lens_size / 2, pos[1] - self.lens_size / 2)
        self.center_point.pos = (pos[0] - 2.5, pos[1] - 2.5)

        img_pos_x, img_pos_y = self.image_rect.pos
        img_width, img_height = self.image_rect.size

        rel_x = (pos[0] - img_pos_x) / img_width
        rel_y = (pos[1] - img_pos_y) / img_height

        x = rel_x * self.img.width
        y = (1 - rel_y) * self.img.height

        crop_width = self.lens_size / self.zoom_level
        crop_height = self.lens_size / self.zoom_level

        left = max(0, min(x - crop_width / 2, self.img.width - crop_width))
        top = max(0, min(y - crop_height / 2, self.img.height - crop_height))
        right = left + crop_width
        bottom = top + crop_height

        black_image = PilImage.new('RGB', (self.lens_size, self.lens_size), (0, 0, 0))

        overlap_x1 = max(0, img_pos_x - (pos[0] - self.lens_size / 2))
        overlap_y1 = max(0, img_pos_y - (pos[1] + self.lens_size / 2))
        overlap_x2 = min(self.lens_size, img_pos_x + img_width - (pos[0] - self.lens_size / 2))
        overlap_y2 = min(self.lens_size, img_pos_y + img_height + (pos[1] - self.lens_size / 2))

        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
            region = self.img.crop((left, top, right, bottom))
            zoomed_region = region.resize((self.lens_size, self.lens_size), PilImage.LANCZOS)
            visible_region = zoomed_region.crop((overlap_x1, overlap_y1, overlap_x2, overlap_y2))
            black_image.paste(visible_region, (int(overlap_x1), int(overlap_y1)))

        self.update_lens(black_image)

    def update_lens(self, pil_image):
        texture = Texture.create(size=(self.lens_size, self.lens_size))
        texture.blit_buffer(pil_image.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()
        self.lens.texture = texture

    def on_mouse_pos(self, *args):
        self.mouse_pos = args[1]

class DraggablePoints(Widget):
    def __init__(self, points, img, image_rect, point_size=20, use_special_magnifier=False, **kwargs):
        super().__init__(**kwargs)
        self.points = points
        self.point_size = point_size
        self.selected_point_index = None
        self.img = img
        self.image_rect = image_rect
        self.magnifier = None
        self.use_special_magnifier = use_special_magnifier
        self.point_canvas = self.canvas
        self.ellipses = []
        self.draw_points_and_lines()

    def draw_points_and_lines(self):
        self.point_canvas.clear()
        with self.point_canvas:
            Color(0, 1, 0)
            self.line = Line(points=[x for point in self.points for x in point[:2]], width=2, close=True)
            Color(1, 0, 0)
            self.ellipses = [Ellipse(pos=(x - self.point_size / 2, y - self.point_size / 2), size=(self.point_size, self.point_size)) for x, y, _ in self.points]
            
    def on_touch_down(self, touch):
        for i, (x, y, idx) in enumerate(self.points):
            if x - self.point_size / 2 <= touch.x <= x + self.point_size / 2 and - self.point_size / 2 <= touch.y <= y + self.point_size / 2:
                self.selected_point_index = i
                self.show_magnifier(touch)
                return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.selected_point_index is not None:
            x, y, idx = self.points[self.selected_point_index]
            self.points[self.selected_point_index] = (touch.x, touch.y, idx)
            self.magnifier.mouse_pos = touch.pos
            self.update_points_and_lines()
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.selected_point_index is not None:
            x, y, idx = self.points[self.selected_point_index]
            self.points[self.selected_point_index] = (touch.x, touch.y, idx)
            self.remove_magnifier()
            self.selected_point_index = None
            self.draw_points_and_lines()
            return True
        return super().on_touch_up(touch)

    def show_magnifier(self, touch):
        if self.magnifier is None:
            self.magnifier = Magnifier(self.img, self.image_rect)
            self.magnifier.mouse_pos = touch.pos
            self.add_widget(self.magnifier)

    def remove_magnifier(self):
        if self.magnifier:
            self.remove_widget(self.magnifier)
            self.magnifier = None

    def update_points_and_lines(self):
        self.line.points = [x for point in self.points for x in point[:2]]
        for idx, (x, y, _) in enumerate(self.points):
            if idx != self.selected_point_index:
                self.ellipses[idx].pos = (x - self.point_size / 2, y - self.point_size / 2)
            else:
                self.ellipses[idx].pos = (-self.point_size, -self.point_size)

class DraggableButton(Button):
    def __init__(self, **kwargs):
        super(DraggableButton, self).__init__(**kwargs)
        self.size_hint = (None, None)
        self.text = ''
        self.background_color = (1, 0, 0, 0.3)
        self.border = (2, 2, 2, 2)
        self.border_color = (0, 0, 0, 1)

        with self.canvas.after:
            Color(0, 0, 0, 1)
            self.line = Line(width=1)

        self.bind(pos=self.update_line, size=self.update_line)
        Clock.schedule_once(self.update_line, 0)

    def update_line(self, *args):
        self.line.points = [self.x, self.center_y, self.right, self.center_y]

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            touch.grab(self)
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            new_y = self.y + touch.dy
            penetration_amount = 1.75 * self.height

            if self.id == 'button1':
                lower_limit = 0
                upper_limit = self.parent.draggable_circle.circle.pos[1] - self.height + penetration_amount
                new_y = min(new_y, upper_limit)
                new_y = max(new_y, lower_limit)
            elif self.id == 'button2':
                lower_limit = self.parent.draggable_circle.circle.pos[1] + self.parent.draggable_circle.circle.size[1] - penetration_amount
                upper_limit = self.parent.height - self.height
                new_y = max(new_y, lower_limit)
                new_y = min(new_y, upper_limit)

            self.y = new_y
            self.parent.update_rf_value()
            self.update_line()
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            return True
        return super().on_touch_up(touch)

class SimpleDraggableCircle(Widget):
    circle_counter = 1

    def __init__(self, table_widget, **kwargs):
        super().__init__(**kwargs)
        self.circle_size = (50, 50)
        self.table_widget = table_widget
        self.circle_id = SimpleDraggableCircle.circle_counter
        SimpleDraggableCircle.circle_counter += 1
        self.label = Label(
            text=f'{self.table_widget.get_index(self) or self.circle_id} (0.00)',
            size_hint=(None, None), 
            font_size=30, 
            color=(1, 1, 1, 1)
        )
        self.update_label_offsets()
        self.draw_circle()
        self.center_on_screen()
        self.double_click = False
        self.double_click_trigger = Clock.create_trigger(self.reset_double_click, 0.3)

    def reset_double_click(self, dt):
        self.double_click = False

    def update_label_offsets(self):
        self.offset_x = 0
        self.offset_y = 0

    def draw_circle(self):
        with self.canvas:
            self.canvas.clear()
            Color(1, 0, 0, 0.5)
            self.circle = Ellipse(size=self.circle_size, pos=self.circle.pos if hasattr(self, 'circle') else (0, 0))
        self.update_label_pos()
        self.update_label_size()

    def center_on_screen(self):
        center_x = Window.width / 2 - self.circle_size[0] / 2
        center_y = Window.height / 2 - self.circle_size[1] / 2
        self.circle.pos = (center_x, center_y)
        self.update_label_pos()

    def set_label_offset(self, offset_x, offset_y):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.update_label_pos()

    def update_label_pos(self):
        new_x, new_y = self.circle.pos
        self.label.center_x = new_x + self.circle.size[0] / 2 + self.offset_x
        self.label.center_y = new_y + self.circle.size[1] / 2 + self.offset_y

    def update_label_size(self):
        self.label.font_size = self.circle_size[0] * 0.28
        self.label.texture_update()  
        self.update_label_pos()

    def on_touch_down(self, touch):
        circle_center = (self.circle.pos[0] + self.circle.size[0] / 2, self.circle.pos[1] + self.circle_size[1] / 2)
        circle_radius = self.circle_size[0] / 2
        distance = ((touch.x - circle_center[0]) ** 2 + (touch.y - circle_center[1]) ** 2) ** 0.5

        if distance <= circle_radius:
            if self.double_click:
                self.double_click = False
                new_circle = SimpleDraggableCircle(self.table_widget)
                new_circle.circle_size = self.circle_size
                new_circle.center_on_screen()
                self.parent.add_widget(new_circle)
                self.parent.add_widget(new_circle.label)
                new_circle.draw_circle()
                new_circle.calculate_rf()
                self.update_all_rf_values()
            else:
                self.double_click = True
                self.double_click_trigger()
            touch.grab(self)
            self.parent.last_dragged_circle = self
            return True
        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            return True
        return super().on_touch_up(touch)

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            image_rect = self.parent.image_rect
            rect_x, rect_y = image_rect.pos
            rect_width, rect_height = image_rect.size

            circle_center_y = self.circle.pos[1] + self.circle_size[1] / 2
            button1_center_y = self.parent.draggable_button1.center_y
            button2_center_y = self.parent.draggable_button2.center_y
            circle_half_height = self.circle_size[1] / 2

            top_limit = button2_center_y + circle_half_height
            bottom_limit = button1_center_y - circle_half_height
            left_limit = rect_x
            right_limit = rect_x + rect_width - self.circle_size[0]

            new_x = min(max(touch.x - self.circle_size[0] / 2, left_limit), right_limit)
            new_y = min(max(touch.y - self.circle_size[1] / 2, bottom_limit), top_limit - self.circle_size[1])

            self.circle.pos = (new_x, new_y)
            self.update_label_pos()
            self.calculate_rf()
            self.parent.last_dragged_circle = self

    def calculate_rf(self):
        button1_center_y = self.parent.draggable_button1.center_y
        button2_center_y = self.parent.draggable_button2.center_y
        circle_center_y = self.circle.pos[1] + self.circle.size[1] / 2

        distance_button1_circle = abs(button1_center_y - circle_center_y)
        distance_button1_button2 = abs(button1_center_y - button2_center_y)

        if distance_button1_button2 != 0:
            Rf = distance_button1_circle / distance_button1_button2
            index = self.table_widget.get_index(self)
            if index is not None:
                self.label.text = f'{index} ({Rf:.2f})'
            else:
                self.label.text = f'(Indefinido)'
        else:
            self.label.text = f'{self.circle_id} (Indefinido)'

        if self.table_widget:
            self.table_widget.update_row(self, f'{Rf:.2f}')

    def increase_circle_size(self):
        old_center_x = self.circle.pos[0] + self.circle_size[0] / 2
        old_center_y = self.circle.pos[1] + self.circle_size[1] / 2

        self.circle_size = (self.circle_size[0] + 10, self.circle_size[1] + 10)
        self.circle.pos = (old_center_x - self.circle_size[0] / 2, old_center_y - self.circle_size[1] / 2)

        self.update_label_offsets()
        self.update_label_pos()
        self.update_label_size()
        self.draw_circle()

    def decrease_circle_size(self):
        if self.circle_size[0] > 20 and self.circle_size[1] > 20:
            old_center_x = self.circle.pos[0] + self.circle_size[0] / 2
            old_center_y = self.circle.pos[1] + self.circle_size[1] / 2

            self.circle_size = (self.circle_size[0] - 10, self.circle_size[1] - 10)
            self.circle.pos = (old_center_x - self.circle_size[0] / 2, old_center_y - self.circle_size[1] / 2)

            self.update_label_offsets()
            self.update_label_pos()
            self.update_label_size()
            self.draw_circle()

    def update_all_rf_values(self):
        for widget in self.parent.children:
            if isinstance(widget, SimpleDraggableCircle):
                widget.calculate_rf()

class DraggableWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dragging = False
        self.drag_offset_x = 0
        self.drag_offset_y = 0

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.dragging = True
            self.drag_offset_x = self.x - touch.x
            self.drag_offset_y = self.y - touch.y
            touch.grab(self)
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if touch.grab_current is self and self.dragging:
            self.x = touch.x + self.drag_offset_x
            self.y = touch.y + self.drag_offset_y
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            self.dragging = False
            touch.ungrab(self)
            return True
        return super().on_touch_up(touch)

class RfTable(DraggableWidget, GridLayout):
    def __init__(self, **kwargs):
        super(RfTable, self).__init__(**kwargs)
        self.cols = 2
        self.size_hint = (None, None)
        self.spacing = [2, 2]
        self.padding = [20, 20, 2, 2]
        self.size = (300, 400)
        self.row_default_height = 40
        self.row_force_default = True
        self.col_default_width = 60
        self.col_force_default = True

        self.header_labels = [
            self.create_shadowed_label('Mol.'),
            self.create_shadowed_label('Rf')
        ]

        for label in self.header_labels:
            label.bind(size=self.update_label_shadow)
            label.bind(pos=self.update_label_shadow)
            label.bind(texture_size=label.setter('text_size'))
            label.text_size = label.size
            self.add_widget(label)

        self.circle_map = {}
        self.next_index = 1
        self.deleted_indices = []

    def create_shadowed_label(self, text):
        label = Label(
            text=text,
            color=(0.8, 0.8, 0.8, 1),
            size_hint=(None, None),
            halign='center',
            valign='middle',
            font_size=24,
            size=(self.col_default_width, self.row_default_height)
        )
        label.bind(size=self.update_label_shadow)
        label.bind(pos=self.update_label_shadow)
        label.bind(texture_size=label.setter('text_size'))
        label.text_size = label.size
        return label

    def update_label_shadow(self, instance, value):
        instance.canvas.before.clear()
        with instance.canvas.before:
            Color(0, 0, 0, 0.5)
            Rectangle(pos=(instance.x - 1, instance.y - 1), size=instance.size)

    def add_row(self, rf_value):
        if self.deleted_indices:
            index = self.next_index
            self.next_index += 1
            self.deleted_indices.remove(min(self.deleted_indices))
        else:
            index = self.next_index
            self.next_index += 1

        index_label = self.create_shadowed_label(str(index))
        rf_label = self.create_shadowed_label(rf_value)

        self.add_widget(index_label)
        self.add_widget(rf_label)
        return index_label, rf_label

    def update_row(self, circle, rf_value):
        if circle not in self.circle_map:
            index_label, rf_label = self.add_row(rf_value)
            self.circle_map[circle] = (index_label, rf_label)
        else:
            _, rf_label = self.circle_map[circle]
            rf_label.text = rf_value

    def remove_row(self, circle):
        index = self.get_index(circle)
        if index == 1:
            return
        if circle in self.circle_map:
            index_label, rf_label = self.circle_map.pop(circle)
            self.remove_widget(index_label)
            self.remove_widget(rf_label)
            self.deleted_indices.append(int(index_label.text))
            self.update_indices()

    def update_indices(self):
        current_index = 1
        for idx, (key, value) in enumerate(sorted(self.circle_map.items(), key=lambda x: int(x[1][0].text)), start=1):
            index_label, rf_label = value
            index_label.text = str(current_index)
            key.label.text = f'{current_index} ({key.label.text.split("(")[1]}'
            current_index += 1
        self.next_index = current_index

    def clear_table(self):
        self.clear_widgets()
        self.cols = 2
        for label in self.header_labels:
            self.add_widget(label)
        self.circle_map = {}
        self.next_index = 1
        self.deleted_indices = []

    def get_index(self, circle):
        for idx, (key, value) in enumerate(self.circle_map.items(), start=1):
            if key == circle:
                return int(value[0].text)
        return None

class ProcessedImage(FloatLayout):
    screen_counter = 0

    def __init__(self, num_points=1, **kwargs):
        super().__init__(**kwargs)
        ProcessedImage.screen_counter += 1
        self.screen_id = ProcessedImage.screen_counter

        self.image_rect = None
        self.load_btn = MDRaisedButton(text='Load Image', size_hint=(0.5, 0.1), pos_hint={'top': 1, 'x': 0})
        self.load_btn.bind(on_release=self.load_image)
        self.add_widget(self.load_btn)

        self.camera_btn = MDRaisedButton(text='Take picture', size_hint=(0.5, 0.1), pos_hint={'top': 1, 'x': 0.5})
        self.camera_btn.bind(on_release=self.capture_image)
        self.add_widget(self.camera_btn)

        self.image_loaded = False
        self.processed_img = None
        self.num_points = num_points
        self.processed_with_canny = False
        self.manual_processing_enabled = False
        self.last_dragged_circle = None
        self.rf_table = None

    def initialize_rf_value(self):
        button1_center_y = self.draggable_button1.center_y
        button2_center_y = self.draggable_button2.center_y
        circle_center_y = (button1_center_y + button2_center_y) / 2
        new_pos = (self.draggable_circle.circle.pos[0], circle_center_y - self.draggable_circle.circle.size[1] / 2)
        self.draggable_circle.circle.pos = new_pos
        self.update_rf_value()

    def update_rf_value(self):
        if hasattr(self, 'draggable_button1') and hasattr(self, 'draggable_button2') and hasattr(self, 'draggable_circle'):
            button1_center_y = self.draggable_button1.center_y
            button2_center_y = self.draggable_button2.center_y
            circle_center_y = self.draggable_circle.circle.pos[1] + self.draggable_circle.circle.size[1] / 2
            distance_button1_circle = abs(circle_center_y - button1_center_y)
            distance_button1_button2 = abs(button2_center_y - button1_center_y)
            if distance_button1_button2 != 0:
                Rf = distance_button1_circle / distance_button1_button2
                self.draggable_circle.label.text = f'{self.draggable_circle.circle_id} ({Rf:.2f})'
            else:
                self.draggable_circle.label.text = f'{self.draggable_circle.circle_id} (Indefinido)'
                
            if self.rf_table:
                self.rf_table.update_row(self.draggable_circle, f'{Rf:.2f}')

    def update_all_rf_values(self):
        for widget in self.children:
            if isinstance(widget, SimpleDraggableCircle):
                widget.calculate_rf()

    def add_image(self, filepath):
        try:
            filepath = normalize_path(filepath)
            if not os.path.exists(filepath):
                return
            
            with open(filepath, 'rb') as file:
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                return

            self.processed_img, self.processed_with_canny = self.process_image(img)
            save_image(self.processed_img, 'processed_image.png')  # Usar la nueva función save_image
            self.display_image(os.path.join(temp_dir, 'processed_image.png'), self.processed_img.shape[1], self.processed_img.shape[0])

            self.load_btn.opacity = 0
            self.load_btn.disabled = True
            self.camera_btn.opacity = 0
            self.camera_btn.disabled = True
            self.image_loaded = True

            if self.processed_with_canny:
                self.manual_processing_enabled = False
                self.clear_table_and_points()
                self.add_rf_table()
                self.add_draggable_buttons(self.image_rect.size[0])
                self.add_size_buttons()
                self.add_manual_button()
            else:
                self.clear_table_and_points()
                self.detect_corners_and_add_points()
                self.manual_processing_enabled = True
                self.add_manual_process_button()
        except Exception as e:
            pass

    def process_manual_callback(self, instance):
        if self.manual_processing_enabled and hasattr(self, 'draggable_points'):
            corners = self.draggable_points.points
            img_pos_x, img_pos_y = self.image_rect.pos
            img_width, img_height = self.image_rect.size
            original_width, original_height = self.original_image.shape[1], self.original_image.shape[0]
            scale_x = original_width / img_width
            scale_y = original_height / img_height

            adjusted_corners = []
            for x, y, _ in corners:
                adjusted_x = (x - img_pos_x) * scale_x
                adjusted_y = (y - img_pos_y) * scale_y
                adjusted_y = original_height - adjusted_y
                adjusted_corners.append((int(adjusted_x), int(adjusted_y)))

            adjusted_corners = np.array(adjusted_corners)
            adjusted_corners = order_points(adjusted_corners)
            dest = find_dest(adjusted_corners)

            try:
                M = cv2.getPerspectiveTransform(np.float32(adjusted_corners), np.float32(dest))
                final = cv2.warpPerspective(self.original_image, M, (dest[2][0], dest[2][1]))

                final_flipped = final
                save_image(final_flipped, 'processed_image_manual.png')  # Usar la nueva función save_image
                self.display_image(os.path.join(temp_dir, 'processed_image_manual.png'), final_flipped.shape[1], final_flipped.shape[0])

                self.processed_img = final_flipped
                self.manual_processing_enabled = False
                self.remove_widget(self.process_btn)
                self.remove_draggable_points()
                self.add_rf_table()
                self.add_draggable_buttons(self.image_rect.size[0])
                self.add_size_buttons()
                self.update_rf_value()
                self.reset_labels_and_rf()
            except cv2.error as e:
                pass

    def handle_manual_mode(self):
        self.clear_table_and_points()
        self.detect_corners_and_add_points()
        self.manual_processing_enabled = True
        self.add_manual_process_button()

    def clear_table_and_points(self):
        if self.rf_table:
            self.rf_table.clear_table()
        for widget in self.children:
            if isinstance(widget, SimpleDraggableCircle):
                self.remove_widget(widget)
                self.remove_widget(widget.label)

    def display_image(self, img_path, img_width, img_height):
        self.canvas.clear()
        layout_width, layout_height = self.size
        img_aspect_ratio = img_width / img_height
        layout_aspect_ratio = layout_width / layout_height

        if img_aspect_ratio > layout_aspect_ratio:
            display_width = layout_width
            display_height = layout_width / img_aspect_ratio
        else:
            display_height = layout_height
            display_width = layout_height * img_aspect_ratio

        pos_x = (layout_width - display_width) / 2
        pos_y = (layout_height - display_height) / 2

        with self.canvas:
            self.image_rect = Rectangle(source=img_path, pos=(pos_x, pos_y), size=(display_width, display_height))

    def process_image(self, img):
        dim_limit = 1080
        max_dim = max(img.shape[:2])
        if max_dim > dim_limit:
            scale = dim_limit / max_dim
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        self.original_image = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for c in contours:
            peri = cv2.arcLength(c, True)
            corners = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(corners) == 4:
                corners = order_points(np.concatenate(corners).tolist())
                dest = find_dest(corners)
                M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(dest))
                final = cv2.warpPerspective(self.original_image, M, (dest[2][0], dest[2][1]))
                self.reset_labels_and_rf()
                return final, True
        self.reset_labels_and_rf()
        return img, False

    def reset_labels_and_rf(self):
        SimpleDraggableCircle.circle_counter = 1
        for widget in self.children:
            if isinstance(widget, SimpleDraggableCircle):
                widget.circle_id = SimpleDraggableCircle.circle_counter
                SimpleDraggableCircle.circle_counter += 1
                index = widget.table_widget.get_index(widget)
                widget.label.text = f'{index or widget.circle_id} (0.00)'
                widget.label.canvas.ask_update()
                widget.calculate_rf()

    def load_image(self, instance):
        if platform == 'android':
            filechooser.open_file(on_selection=self.handle_image_selection)
        else:
            self.load_image_from_desktop()

    def load_image_from_desktop(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        if file_path:
            normalized_path = normalize_path(file_path)
            if os.path.exists(normalized_path):
                self.add_image(normalized_path)
        root.destroy()

    def handle_image_selection(self, selection):
        if selection:
            normalized_path = normalize_path(selection[0])
            if os.path.exists(normalized_path):
                self.add_image(normalized_path)

    def capture_image(self, instance):
        if platform == 'android':
            camera.take_picture(on_complete=self.handle_camera_capture, filename='camera_image.jpg')
        else:
            self.show_popup()

    def show_popup(self):
        content = FloatLayout()
        label = Label(text="Próximamente", font_size='20sp', size_hint=(None, None), size=(200, 50), pos_hint={'center_x': 0.5, 'center_y': 0.6})
        btn = Button(text='Cerrar', size_hint=(None, None), size=(100, 50), pos_hint={'center_x': 0.5, 'y': 0})
        content.add_widget(label)
        content.add_widget(btn)

        popup = Popup(title='Información', content=content, size_hint=(None, None), size=(300, 200))
        btn.bind(on_release=popup.dismiss)
        popup.open()

    def handle_camera_capture(self, path):
        if path:
            normalized_path = normalize_path(path)
            self.add_image(normalized_path)

    def detect_corners_and_add_points(self):
        if self.image_rect:
            img_pos_x, img_pos_y = self.image_rect.pos
            img_width, img_height = self.image_rect.size

            img = PilImage.open(os.path.join(temp_dir, "processed_image.png"))
            points = [
                (img_pos_x + img_width * 0.1, img_pos_y + img_height * 0.1),
                (img_pos_x + img_width * 0.9, img_pos_y + img_height * 0.1),
                (img_pos_x + img_width * 0.9, img_pos_y + img_height * 0.9),
                (img_pos_x + img_width * 0.1, img_pos_y + img_height * 0.9)
            ]
            indexed_points = [(x, y, idx + 1) for idx, (x, y) in enumerate(points)]
            self.draggable_points = DraggablePoints(indexed_points, img, self.image_rect, use_special_magnifier=False)
            self.add_widget(self.draggable_points)

            if self.rf_table:
                self.remove_widget(self.rf_table)
                self.rf_table = None

    def detect_corners_and_add_points_special(self):
        if self.image_rect:
            img_pos_x, img_pos_y = self.image_rect.pos
            img_width, img_height = self.image_rect.size

            img = PilImage.open(os.path.join(temp_dir, "original_image.png"))
            points = [
                (img_pos_x + img_width * 0.1, img_pos_y + img_height * 0.1),
                (img_pos_x + img_width * 0.9, img_pos_y + img_height * 0.1),
                (img_pos_x + img_width * 0.9, img_pos_y + img_height * 0.9),
                (img_pos_x + img_width * 0.1, img_pos_y + img_height * 0.9)
            ]
            indexed_points = [(x, y, idx + 1) for idx, (x, y) in enumerate(points)]
            self.draggable_points = DraggablePoints(indexed_points, img, self.image_rect, use_special_magnifier=False)
            self.add_widget(self.draggable_points)

            if self.rf_table:
                self.remove_widget(self.rf_table)
                self.rf_table = None

    def remove_draggable_points(self):
        if hasattr(self, 'draggable_points'):
            self.remove_widget(self.draggable_points)

    def add_rf_table(self):
        if self.rf_table is None:
            self.rf_table = RfTable(size_hint=(None, None), size=(300, 400), pos=(0, self.height - 400))
            self.add_widget(self.rf_table)

    def add_draggable_buttons(self, image_width):
        image_width = self.image_rect.size[0]
        image_x = self.image_rect.pos[0]

        self.draggable_button1 = DraggableButton(size_hint=(None, None), size=(image_width, 20))
        self.draggable_button1.pos = (image_x, 40)
        self.draggable_button1.id = 'button1'
        self.draggable_button1.bind(pos=self.on_button_move)
        self.add_widget(self.draggable_button1)

        self.draggable_button2 = DraggableButton(size_hint=(None, None), size=(image_width, 20))
        self.draggable_button2.pos = (image_x, self.height - 70)
        self.draggable_button2.id = 'button2'
        self.draggable_button2.bind(pos=self.on_button_move)
        self.add_widget(self.draggable_button2)

        self.draggable_circle = SimpleDraggableCircle(self.rf_table, pos=(0, 0), size=(self.width, self.height))
        self.add_widget(self.draggable_circle)
        self.add_widget(self.draggable_circle.label)
        self.draggable_circle.center_on_screen()
        self.last_dragged_circle = self.draggable_circle
        self.initialize_rf_value()

    def add_manual_process_button(self):
        self.process_btn = MDRaisedButton(text='Procesar Manualmente', size_hint=(0.1, 0.08), pos_hint={'right': 1, 'y': 0})
        self.process_btn.bind(on_release=self.process_manual_callback)
        self.add_widget(self.process_btn)

    def add_manual_button(self):
        self.manual_btn = MDRaisedButton(text='Manual', size_hint=(0.1, 0.1), pos_hint={'x': 0.9, 'top': 1})
        self.manual_btn.bind(on_release=self.switch_to_manual_mode)
        self.add_widget(self.manual_btn)

    def switch_to_manual_mode(self, instance):
        if self.original_image is None:
            return
        try:
            self.manual_processing_enabled = True
            self.processed_with_canny = False
            original_image_path = os.path.join(temp_dir, 'original_image.png')
            save_image(self.original_image, 'original_image.png')  # Usar la nueva función save_image

            self.display_image_fullscreen(original_image_path, self.original_image.shape[1], self.original_image.shape[0])
            self.detect_corners_and_add_points_special()
            self.add_manual_process_button()
            self.reset_labels_and_rf()
        except Exception as e:
            pass

    def display_image_fullscreen(self, img_path, img_width, img_height):
        self.canvas.clear()
        layout_width, layout_height = self.size
        img_aspect_ratio = img_width / img_height
        layout_aspect_ratio = layout_width / layout_height

        if img_aspect_ratio > layout_aspect_ratio:
            display_width = layout_width
            display_height = layout_width / img_aspect_ratio
        else:
            display_height = layout_height
            display_width = layout_height * img_aspect_ratio

        pos_x = (layout_width - display_width) / 2
        pos_y = (layout_height - display_height) / 2

        with self.canvas:
            self.image_rect = Rectangle(source=img_path, pos=(pos_x, pos_y), size=(display_width, display_height))

    def add_size_buttons(self):
        button_layout = BoxLayout(
            orientation='vertical',
            spacing=5,
            size_hint=(0.15, 0.4),
            pos_hint={'right': 1.1, 'top': 0.4}
        )
        button_height = 40
        self.save_btn = MDRaisedButton(text="Save", size_hint_y=None, height=button_height)
        self.save_btn.bind(on_release=self.show_save_dialog)
        button_layout.add_widget(self.save_btn)

        self.toggle_rf_btn = MDRaisedButton(text='Hide', size_hint_y=None, height=button_height)
        self.toggle_rf_btn.bind(on_release=self.toggle_rf_visibility)
        button_layout.add_widget(self.toggle_rf_btn)

        self.delete_last_point_btn = MDRaisedButton(text='del', size_hint_y=None, height=button_height)
        self.delete_last_point_btn.bind(on_release=self.delete_last_point)
        button_layout.add_widget(self.delete_last_point_btn)

        self.increase_size_btn = MDRaisedButton(text="+", size_hint_y=None, height=button_height)
        self.increase_size_btn.bind(on_release=lambda instance: self.adjust_circle_size(increase=True))
        button_layout.add_widget(self.increase_size_btn)

        self.decrease_size_btn = MDRaisedButton(text="-", size_hint_y=None, height=button_height)
        self.decrease_size_btn.bind(on_release=lambda instance: self.adjust_circle_size(increase=False))
        button_layout.add_widget(self.decrease_size_btn)

        self.home_btn = MDRaisedButton(text='Exit', size_hint_y=None, height=button_height)
        self.home_btn.bind(on_release=self.go_home)
        button_layout.add_widget(self.home_btn)

        self.add_widget(button_layout)

        spinner_layout = BoxLayout(
            orientation='vertical',
            spacing=5,
            size_hint=(0.15, 0.2),
            pos_hint={'right': 1, 'top': 0.6}
        )
        self.spinner1 = Spinner(text='Solvent', values=('Hexane', 'AcOEt', 'DCM', 'MeOH', 'Ethanol', 'Acetonitrile', 'Benzene', 'Chloroform', 'Acetone', 'AcOH', 'c-hexane', 'Toluene', 'None'))
        self.spinner3 = Spinner(text='0', values=[str(i) for i in range(0, 10)])
        self.spinner2 = Spinner(text='Solvent', values=('Hexane', 'AcOEt', 'DCM', 'MeOH', 'Ethanol', 'Acetonitrile', 'Benzene', 'Chloroform', 'Acetone', 'AcOH', 'c-hexane', 'Toluene', 'None'))
        self.spinner4 = Spinner(text='0', values=[str(i) for i in range(0, 10)])

        first_row_layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=30)
        first_row_layout.add_widget(self.spinner1)
        first_row_layout.add_widget(self.spinner3)

        second_row_layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=30)
        second_row_layout.add_widget(self.spinner2)
        second_row_layout.add_widget(self.spinner4)

        spinner_layout.add_widget(first_row_layout)
        spinner_layout.add_widget(second_row_layout)
        self.add_widget(spinner_layout)

    def delete_last_point(self, *args):
        if self.last_dragged_circle:
            index = self.rf_table.get_index(self.last_dragged_circle)
            if index == 1:
                return
            self.remove_widget(self.last_dragged_circle)
            self.remove_widget(self.last_dragged_circle.label)
            if self.rf_table:
                self.rf_table.remove_row(self.last_dragged_circle)
            self.last_dragged_circle = None

    def show_save_dialog(self, *args):
        content = BoxLayout(orientation='vertical')
        self.save_text_input = TextInput(hint_text='Enter filename', multiline=False)
        self.save_path_input = TextInput(hint_text='Enter save path', multiline=False)
        select_path_button = Button(text='Select Path', size_hint=(1, 0.3))
        select_path_button.bind(on_release=self.select_save_path)
        self.format_spinner = Spinner(text='jpg', values=('jpg', 'bmp'))

        save_button = Button(text='Save', size_hint=(1, 0.3))
        save_button.bind(on_release=self.save_screen)
        content.add_widget(self.save_text_input)
        content.add_widget(self.save_path_input)
        content.add_widget(select_path_button)
        content.add_widget(self.format_spinner)
        content.add_widget(save_button)

        self.save_popup = Popup(title='Save Image', content=content, size_hint=(0.7, 0.7))
        self.save_popup.open()

    def select_save_path(self, *args):
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory()
        if path:
            self.save_path_input.text = path
        root.destroy()

    def save_screen(self, *args):
        buttons_to_hide = [
            self.save_btn,
            self.toggle_rf_btn,
            self.delete_last_point_btn,
            self.increase_size_btn,
            self.decrease_size_btn,
            self.home_btn
        ]
        if hasattr(self, 'manual_btn'):
            buttons_to_hide.append(self.manual_btn)
        for btn in buttons_to_hide:
            btn.opacity = 0
            btn.disabled = True

        try:
            with self.canvas:
                capture_group = InstructionGroup()
                self.canvas.add(capture_group)

                for widget in self.children:
                    if isinstance(widget, (ProcessedImage, RfTable)):
                        capture_group.add(widget.canvas)

            filename = self.save_text_input.text.strip()
            save_path = self.save_path_input.text.strip()
            file_format = self.format_spinner.text.lower()

            if not filename:
                filename = "saved_screen"
            if not filename.lower().endswith(f".{file_format}"):
                filename += f".{file_format}"

            self.save_popup.dismiss()
            if not save_path:
                save_path = os.path.expanduser("~")

            full_path = os.path.join(save_path, filename)
            if not os.path.exists(save_path):
                try:
                    os.makedirs(save_path)
                except Exception as e:
                    print(f"Error al crear el directorio: {e}")
                    return

            try:
                temp_path = os.path.join(temp_dir, "temp_screenshot.png")
                self.export_to_png(temp_path)
                img = PilImage.open(temp_path)

                if img is None:
                    print("Error: No se pudo cargar la imagen temporal.")
                else:
                    if file_format == 'jpg':
                        file_format = 'JPEG'
                        if img.mode == 'RGBA':
                            img = img.convert('RGB')
                        img.save(full_path, file_format)
                    else:
                        file_format = file_format.upper()
                        img.save(full_path, file_format)

                    print(f"La imagen ha sido guardada como '{full_path}'.")
                os.remove(temp_path)
            except Exception as e:
                print(f"Error al guardar la imagen: {e}")
            finally:
                self.canvas.remove(capture_group)
        finally:
            for btn in buttons_to_hide:
                btn.opacity = 1
                btn.disabled = False

    def toggle_rf_visibility(self, instance):
        for circle in self.rf_table.circle_map.keys():
            circle.label.opacity = 0 if circle.label.opacity == 1 else 1

    def adjust_circle_size(self, increase):
        if self.last_dragged_circle:
            if increase:
                self.last_dragged_circle.increase_circle_size()
            else:
                self.last_dragged_circle.decrease_circle_size()

    def remove_draggable_buttons(self):
        if hasattr(self, 'draggable_button1'):
            self.remove_widget(self.draggable_button1)
        if hasattr(self, 'draggable_button2'):
            self.remove_widget(self.draggable_button2)
        if hasattr(self, 'draggable_circle'):
            self.remove_widget(self.draggable_circle)
        if hasattr(self, 'increase_size_btn'):
            self.remove_widget(self.increase_size_btn)
        if hasattr(self, 'decrease_size_btn'):
            self.remove_widget(self.decrease_size_btn)

    def on_button_move(self, instance, value):
        if hasattr(self, 'draggable_button1') and hasattr(self, 'draggable_button2'):
            penetration_amount = 1.75 * instance.height
            highest_circle_top = float('-inf')
            lowest_circle_bottom = float('inf')

            for widget in self.children:
                if isinstance(widget, SimpleDraggableCircle):
                    circle_center_y = widget.circle.pos[1] + widget.circle.size[1] / 2
                    circle_half_height = widget.circle.size[1] / 2
                    top_limit = circle_center_y + circle_half_height - penetration_amount
                    bottom_limit = circle_center_y + circle_half_height - penetration_amount
                    if top_limit > highest_circle_top:
                        highest_circle_top = top_limit
                    if bottom_limit < lowest_circle_bottom:
                        lowest_circle_bottom = bottom_limit

            if instance.id == 'button1':
                instance.y = min(instance.y, lowest_circle_bottom)
            elif instance.id == 'button2':
                instance.y = max(instance.y, highest_circle_top)

            self.update_all_rf_values()

    def go_home(self, instance):
        App.get_running_app().stop()

    def kill_process_using_file(file_path):
        for proc in psutil.process_iter(['pid', 'name', 'open_files']):
            try:
                for file in proc.info['open_files'] or []:
                    if file.path == file_path:
                        proc.terminate()
                        proc.wait()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def delete_temp_images(self):
        temp_images = [os.path.join(temp_dir, 'original_image.png'), 
                       os.path.join(temp_dir, 'processed_image.png'), 
                       os.path.join(temp_dir, 'processed_image_manual.png')]
        for image in temp_images:
            try:
                if os.path.exists(image):
                    kill_process_using_file(image)
                    os.remove(image)
            except Exception as e:
                pass

class IntroScreen(Screen):
    def __init__(self, **kwargs):
        super(IntroScreen, self).__init__(**kwargs)
        layout = FloatLayout()

        # Crear un widget Image sin allow_stretch y keep_ratio
        intro_image = Image(source=image_path)
        intro_image.size_hint = (1, 1)  # Ajusta el tamaño para ocupar todo el espacio disponible
        intro_image.keep_data = True  # Para asegurarnos de que la imagen se mantenga en la memoria

        # Ajustar el tamaño y la posición de la imagen para que se mantenga la relación de aspecto
        layout.bind(size=self.update_image_size)
        layout.add_widget(intro_image)
        self.intro_image = intro_image

        self.add_widget(layout)
        Clock.schedule_once(self.switch_to_main_screen, 4)

    def update_image_size(self, instance, value):
        # Calcular el tamaño de la imagen para que mantenga su relación de aspecto
        layout_width, layout_height = instance.size
        img_width, img_height = self.intro_image.texture.size
        aspect_ratio = img_width / img_height

        if layout_width / layout_height > aspect_ratio:
            # Si el ancho es mayor en proporción, ajustamos la altura
            new_height = layout_height
            new_width = aspect_ratio * new_height
        else:
            # Si la altura es mayor en proporción, ajustamos el ancho
            new_width = layout_width
            new_height = new_width / aspect_ratio

        self.intro_image.size = (new_width, new_height)
        self.intro_image.pos = (
            (layout_width - new_width) / 2,
            (layout_height - new_height) / 2
        )

    def switch_to_main_screen(self, dt):
        self.manager.current = 'main'


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.image_layout = FloatLayout()
        self.add_widget(self.image_layout)
        self.processed_image_widget = ProcessedImage()
        self.image_layout.add_widget(self.processed_image_widget)

class MainApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Amber"
        self.title = 'T.L.C meter'

        sm = ScreenManager()
        sm.add_widget(IntroScreen(name='intro'))
        sm.add_widget(MainScreen(name='main'))
        sm.current = 'intro'
        return sm

    def on_start(self):
        Clock.schedule_once(self.switch_to_main, 4)
        Window.show()  # Mostrar la ventana si está oculta
        Window.maximize()  # Maximizar la ventana si es necesario

    def switch_to_main(self, dt):
        self.root.current = 'main'

if __name__ == "__main__":
    MainApp().run()
