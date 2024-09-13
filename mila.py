import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
from pydub import AudioSegment
import simpleaudio as sa
import librosa
import pyrubberband
import soundfile
import time
import pyqtgraph as pg

from utils import *


annotations_file = "jamendolyrics++/annotations/words/Killing_Time_on_Broadway_-_Marwood_Williams.csv"
slowed = False
# load only a chunk/slice of lyrics to reduce lag
loaded_lyrics = (295, 450)  # (0, 150), (140, 290), (280, 430), (420, 570)

class Audioplayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('MILA')
        self.setGeometry(100, 100, 1200, 600)

        # layout and central widget
        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)

        # play/pause button
        self.play_pause_button = QPushButton('play', self)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.layout.addWidget(self.play_pause_button)

        # save timestamps button
        self.save_timestamps_button = QPushButton('save timestamps', self)
        self.save_timestamps_button.clicked.connect(self.save_timestamps)
        self.layout.addWidget(self.save_timestamps_button)

        # pyqtgraph plot widget for navigation
        self.navigation_widget = pg.PlotWidget()
        self.layout.addWidget(self.navigation_widget, stretch=1)
        self.disable_mouse_interaction_and_hide_y_axis(self.navigation_widget)
        self.navigation_widget.scene().sigMouseClicked.connect(self.on_navigation_click)

        # pyqtgraph plot widget for waveform slice
        self.wave_slice_widget = pg.PlotWidget()
        self.layout.addWidget(self.wave_slice_widget, stretch=5)
        self.disable_mouse_interaction_and_hide_y_axis(self.wave_slice_widget)
        self.wave_slice_widget.scene().sigMouseClicked.connect(self.on_wave_slice_click)

        # create playback line for wave_slice_widget
        self.playback_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', style=pg.QtCore.Qt.DashLine))
        self.wave_slice_widget.addItem(self.playback_line)

        # pyqtgraph plot widget for lyrics
        self.lyrics_slice_widget = pg.PlotWidget()
        self.layout.addWidget(self.lyrics_slice_widget, stretch=1)
        self.disable_mouse_interaction_and_hide_y_axis(self.lyrics_slice_widget)

        # create playback line for lyrics_slice_widget
        self.lyrics_playback_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', style=pg.QtCore.Qt.DashLine))
        self.lyrics_slice_widget.addItem(self.lyrics_playback_line)

        # timer for updating playback lines
        self.timer = QTimer(self)
        self.timer.setInterval(20)  # update every 20 ms (50 fps)
        self.timer.timeout.connect(self.update_playback_line)

        # files
        self.mp3_file = 'jamendolyrics++/mp3/' + annotations_file[39:-4] + '.mp3'
        print(self.mp3_file)
        self.lyrics_file = 'jamendolyrics++/lyrics/' + annotations_file[39:-4] + '.txt'
        self.timestamps_file = annotations_file

        # variables
        self.abs_time_play = 0
        self.time_pause = 0
        self.is_playing = False
        self.play_obj = None  # handle for the audio playback
        self.playback_speed = 0.7 if slowed else 1
        self.wave_slice_start = 0
        self.wave_slice_length = 7
        self.downsample = 40  # downsample the waveform for faster plotting
        self.lyrics_boxes = []

        self.load_audio()
        self.load_lyrics()
        self.init_widgets()

        self.lyrics_slice_widget.scene().sigMouseClicked.connect(self.on_lyrics_click)

    def on_lyrics_click(self, event):
        if event.button() != Qt.LeftButton:
            return  # Only handle left-clicks

        pos = event.scenePos()
        mouse_point = self.lyrics_slice_widget.plotItem.vb.mapSceneToView(pos)
        x = mouse_point.x()

        # Iterate through lyrics boxes to find which one was clicked
        for i, (rect, text_item) in enumerate(self.lyrics_boxes):
            rect_start = rect.pos().x()
            rect_end = rect_start + rect.size().x()

            if rect_start <= x <= rect_end:
                # Found the clicked box
                if i < len(self.lyrics_boxes) - 1:
                    next_rect = self.lyrics_boxes[i + 1][0]
                    next_start = next_rect.pos().x()

                    # Update the current box's end to match the next box's start
                    new_size = next_start - rect_start
                    if new_size <= 0:
                        print("Cannot set end before or equal to start. Operation skipped.")
                        return

                    rect.setSize([new_size, rect.size().y()])
                else:
                    rect.setSize([2, rect.size().y()])

                break  # Exit after handling the clicked box

    def disable_mouse_interaction_and_hide_y_axis(self, widget):
        widget.setMouseEnabled(x=False, y=False)
        widget.getViewBox().setLimits(xMin=0, xMax=None, yMin=-1, yMax=1)
        widget.getViewBox().enableAutoRange(axis=None)
        widget.getPlotItem().hideAxis('left')

    def load_audio(self):
        # slow down audio
        slowed_audio_file = 'slowed_audio.wav'
        y, sr = librosa.load(self.mp3_file, sr=None)
        y_stretched = pyrubberband.time_stretch(y, sr, self.playback_speed)
        soundfile.write(slowed_audio_file, y_stretched, sr, format='wav')

        # load audio data for visualization
        self.waveform, self.sr = librosa.load(slowed_audio_file, sr=None)
        self.audio_length = librosa.get_duration(y=self.waveform, sr=self.sr)

        # load audio with pydub for playback
        self.audio = AudioSegment.from_file(slowed_audio_file)

    def load_lyrics(self):
        with open(self.lyrics_file, 'r') as f:
            lines = f.read().splitlines()
        lines = normalize_jamendo(lines)
        self.words = ' '.join(lines).split()
        self.times = read_jamendo_times(self.timestamps_file)

        if slowed:
            # when aligning load only subset of lyrics to reduce lag
            words = self.words[slice(*loaded_lyrics)]
            print(words)
            times = self.times[slice(*loaded_lyrics)]
        else:
            # when checking at 1x the speed a bit of lag is fine
            words = self.words
            times = self.times

        for i, (word, (start, end)) in enumerate(zip(words, times)):
            self.add_lyrics_box(loaded_lyrics[0] + i, word, start / self.playback_speed, end / self.playback_speed)

    def add_lyrics_box(self, idx, word, start, end):
        rect = pg.RectROI([start, -0.5], [end-start, 1], pen=pg.mkPen('w', width=2), movable=True, resizable=True)
        rect.addScaleHandle([0, 0.5], [1, 0.5])  # left side
        rect.addScaleHandle([1, 0.5], [0, 0.5])  # right side
        self.lyrics_slice_widget.addItem(rect)

        # add text within the box
        text_item = pg.TextItem(f'{idx + 2}\n{word}' if slowed else word, color='w', anchor=(0.5, 0.5))  # +2 to match the index in the annotation.csv file
        self.update_lyrics_text_position(rect, text_item)
        self.lyrics_slice_widget.addItem(text_item)

        # restrict vertical movement by locking y-axis
        rect.sigRegionChanged.connect(self.restrict_vertical_movement)
        # connect the rectangle's signals to update the text position when the rectangle is moved or resized
        rect.sigRegionChanged.connect(lambda: self.update_lyrics_text_position(rect, text_item))

        self.lyrics_boxes.append((rect, text_item))

    def restrict_vertical_movement(self, rect):
        # temporarily disconnect the signal to avoid recursion
        rect.sigRegionChanged.disconnect(self.restrict_vertical_movement)

        # restrict the lyrics rectangle to move only horizontally
        pos = rect.pos()
        rect.setPos([pos.x(), -0.5])

        # reconnect the signal after adjusting the position
        rect.sigRegionChanged.connect(self.restrict_vertical_movement)

    def update_lyrics_text_position(self, rect, text_item):
        rect_center = rect.pos() + rect.size() / 2
        text_item.setPos(rect_center.x(), 0)

    def init_widgets(self):
        # displays full-length audio for easy navigation
        waveform = self.waveform[::self.downsample]
        t = np.linspace(0, self.audio_length, len(waveform))

        self.navigation_widget.clear()
        self.navigation_widget.plot(t, waveform, pen='b')

        # rectangle indicating the currently displayed waveform slice
        self.navigation_rect = pg.RectROI([0, -1], [self.wave_slice_length, 2], pen=pg.mkPen('g', width=1), resizable=False)
        self.navigation_widget.addItem(self.navigation_rect)

        self.wave_slice_widget.clear()
        self.wave_slice_widget.plot(t, waveform, pen='b')
        self.wave_slice_widget.addItem(self.playback_line)

        self.update_slice()

    def update_slice(self):
        # update wave slice
        self.wave_slice_widget.setXRange(self.wave_slice_start, self.wave_slice_start + self.wave_slice_length)
        
        # update navigation rectangle
        self.navigation_rect.setPos([self.wave_slice_start, -1])
        self.navigation_rect.setSize([self.wave_slice_length, 2])

        # update lyrics slice
        self.lyrics_slice_widget.setXRange(self.wave_slice_start, self.wave_slice_start + self.wave_slice_length)

    def update_playback_line(self):
        if self.is_playing:
            current_time = time.time() - self.abs_time_play
        else:
            current_time = self.time_pause

        if current_time >= self.audio_length:
            self.timer.stop()
            self.is_playing = False
            self.play_pause_button.setText('play')
            current_time = self.audio_length

        # update slice should playback line roll over
        if current_time >= self.wave_slice_start + self.wave_slice_length:
            self.wave_slice_start += self.wave_slice_length
            self.update_slice()

        # update playback line position
        self.playback_line.setValue(current_time)
        self.lyrics_playback_line.setValue(current_time)

    def toggle_play_pause(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        skip_ms = int(self.time_pause * 1000)
        audio_from_paused = self.audio[skip_ms:]
        self.play_obj = sa.play_buffer(
            audio_from_paused.raw_data,
            num_channels=self.audio.channels,
            bytes_per_sample=audio_from_paused.sample_width,
            sample_rate=audio_from_paused.frame_rate
        )
        self.abs_time_play = time.time() - self.time_pause  # continue from where we paused
        self.timer.start()
        self.is_playing = True
        self.play_pause_button.setText('pause')

    def pause(self):
        if self.play_obj:
            self.play_obj.stop()
            self.timer.stop()
            self.time_pause = time.time() - self.abs_time_play
            self.is_playing = False
            self.play_pause_button.setText('play')

    def on_navigation_click(self, event):
        pos = event.scenePos()
        mouse_point = self.navigation_widget.plotItem.vb.mapSceneToView(pos)
        click_time = mouse_point.x()

        if click_time < 0:
            click_time = 0
        elif click_time > self.audio_length:
            click_time = self.audio_length

        # pause audio at click_time
        self.pause()
        self.time_pause = click_time

        # update waveform slice
        self.wave_slice_start = max(0, click_time - self.wave_slice_length / 2)
        self.update_slice()
        self.update_playback_line()

    def on_wave_slice_click(self, event):
        pos = event.scenePos()
        mouse_point = self.wave_slice_widget.plotItem.vb.mapSceneToView(pos)
        click_time = mouse_point.x()

        if click_time < self.wave_slice_start:
            click_time = self.wave_slice_start
        elif click_time > self.wave_slice_start + self.wave_slice_length:
            click_time = self.wave_slice_start + self.wave_slice_length

        # pause audio at click_time
        self.pause()
        self.time_pause = click_time

        # update playback line
        self.update_playback_line()
    
    def closeEvent(self, event):
        if slowed:
            self.save_timestamps()
        event.accept()

    def save_timestamps(self):
        times = self.times
        assert slowed
        assert len(self.lyrics_boxes) <= loaded_lyrics[1] - loaded_lyrics[0], f'{len(self.lyrics_boxes)}, {loaded_lyrics[1] - loaded_lyrics[0]}'
        for i, (rect, _) in zip(range(*loaded_lyrics), self.lyrics_boxes):
            start = rect.pos().x()
            end = start + rect.size().x()
            times[i] = (start * self.playback_speed, end * self.playback_speed)

        with open(self.timestamps_file, 'w') as file:
            file.write('word_start,word_end\n')
            for start, end in times:
                file.write(f'{start:.2f},{end:.2f}\n')
        print('timestamps saved')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_play_pause()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = Audioplayer()
    player.show()
    sys.exit(app.exec_())
