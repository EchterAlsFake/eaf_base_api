import sys


class Callback:
    @classmethod
    def custom_callback(cls, downloaded, total):
        """This is an example of how you can implement the custom callback"""

        percentage = (downloaded / total) * 100
        print(f"Downloaded: {downloaded} bytes / {total} bytes ({percentage:.2f}%)")

    @classmethod
    def text_progress_bar(cls, downloaded, total, title=False):
        bar_length = 50
        filled_length = int(round(bar_length * downloaded / float(total)))
        percents = round(100.0 * downloaded / float(total), 1)
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        if title is False:
            print(f"\r[{bar}] {percents}%", end='')

        else:
            print(f"\r | {title} | -->: [{bar}] {percents}%", end='')

    @staticmethod
    def update_progress(downloaded, total, animation_phase):
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
    def animated_text_progress(cls, downloaded, total_size):
        animation_phase = 0
        while downloaded <= total_size:
            cls.update_progress(downloaded, total_size, animation_phase)
            animation_phase += 1
