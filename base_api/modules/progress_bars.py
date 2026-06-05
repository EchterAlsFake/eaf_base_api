import sys
from typing import Union


class Callback:
    @classmethod
    def custom_callback(cls, downloaded: int, total: int) -> None:
        """This is an example of how you can implement the custom callback"""
        if total and total > 0:
            percentage = (downloaded / total) * 100
            print(f"Downloaded: {downloaded} bytes / {total} bytes ({percentage:.2f}%)")
        else:
            print(f"Downloaded: {downloaded} bytes")

    @classmethod
    def text_progress_bar(cls, downloaded: int, total: int, title: Union[str, bool] = False) -> None:
        if not total or total <= 0:
            # If total is unknown or 0, just show downloaded bytes
            if title is False:
                print(f"\r[{downloaded} bytes downloaded]", end='')
            else:
                print(f"\r | {title} | -->: [{downloaded} bytes downloaded]", end='')
            return
            
        bar_length = 50
        filled_length = int(round(bar_length * downloaded / float(total)))
        percents = round(100.0 * downloaded / float(total), 1)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        if title is False:
            print(f"\r[{bar}] {percents}%", end='')

        else:
            print(f"\r | {title} | -->: [{bar}] {percents}%", end='')

    @staticmethod
    def update_progress(downloaded: int, total: int, animation_phase: int) -> None:
        if not total or total <= 0:
            sys.stdout.write(f"\r[{downloaded} bytes downloaded]")
            sys.stdout.flush()
            return
            
        bar_length = 50
        filled_length = int(round(bar_length * downloaded / float(total)))
        percents = round(100.0 * downloaded / float(total), 1)

        # Animation phases could be anything. Here's a simple example with a rotating bar
        animation_characters = ['|', '/', '-', '\\']
        animation_char = animation_characters[animation_phase % len(animation_characters)]

        bar = '#' * filled_length + animation_char + '-' * (bar_length - filled_length - 1)

        sys.stdout.write(f"\r[{bar}] {percents}%")
        sys.stdout.flush()

    @classmethod
    def animated_text_progress(cls, downloaded: int, total_size: int) -> None:
        animation_phase = 0
        while downloaded <= total_size:
            cls.update_progress(downloaded, total_size, animation_phase)
            animation_phase += 1