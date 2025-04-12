# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Sample code that demonstrates how to use Metavision SDK to visualize events from a live camera or an event file
"""

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
import argparse
import os


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision Simple Viewer sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="data/2025_04_02_16_46_18/events.raw",
        help="Path to input event file (RAW, DAT or HDF5). If not specified, the camera live stream is used. "
        "If it's a camera serial number, it will try to open that camera instead.")
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    # Events iterator on Camera or event file
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=1000)
    height, width = mv_iterator.get_size()  # Camera Geometry
    print(f"Camera Geometry: {width}x{height}")

    # 添加变量追踪事件点的最大最小值
    min_x, max_x = width, 0
    min_y, max_y = height, 0

    # Helper iterator to emulate realtime
    if not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)

    # Window - Graphical User Interface
    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:
        def keyboard_cb(key, scancode, action, mods):
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        # Event Frame Generator
        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height, fps=25,
                                                           palette=ColorPalette.Dark)

        def on_cd_frame_cb(ts, cd_frame):
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        # Process events
        for evs in mv_iterator:
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()
            
            # # 计算当前批次事件的x和y的最大最小值
            # if len(evs) > 0:
            #     current_min_x = evs['x'].min()
            #     current_max_x = evs['x'].max()
            #     current_min_y = evs['y'].min()
            #     current_max_y = evs['y'].max()
                
            #     # 更新全局最大最小值
            #     min_x = min(min_x, current_min_x)
            #     max_x = max(max_x, current_max_x)
            #     min_y = min(min_y, current_min_y)
            #     max_y = max(max_y, current_max_y)
                
            #     # 打印当前的最大差距
            #     x_range = max_x - min_x
            #     y_range = max_y - min_y
            #     print(f"X方向最大差距: {x_range}, Y方向最大差距: {y_range}")

            event_frame_gen.process_events(evs)

            if window.should_close():
                break


if __name__ == "__main__":
    main()
