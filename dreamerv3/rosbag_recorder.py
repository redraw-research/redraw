import subprocess
import os

class ROSBagRecorder:
    def __init__(self):
        self.record_process = None
        self._latest_filename = None

    def start(self, topics_list, filename):
        """
        Starts recording a rosbag.

        :param topics_list: List of topics to record (e.g., ["/topic1", "/topic2"]).
        :param filename: Name of the output bag file (e.g., "output.bag").
        """
        if self.record_process is not None:
            raise RuntimeError("Recording is already in progress. Call stop() before starting a new recording.")

        if not topics_list:
            raise ValueError("The topics_list cannot be empty.")

        if not filename:
            raise ValueError("The filename cannot be empty.")

        filename += ".tmp"

        self._latest_filename = filename
        record_command = ["rosbag", "record", "--lz4", "-O", filename] + topics_list

        try:
            env_vars = os.environ.copy()
            env_vars.update({
                "PYTHONPATH": "/opt/ros/noetic/lib/python3/dist-packages"
            })
            self.record_process = subprocess.Popen(record_command, env=env_vars)
            print(f"Started recording rosbag: {filename}")
        except Exception as e:
            self.record_process = None
            raise RuntimeError(f"Failed to start recording: {e}")

    def stop(self):
        """
        Stops the current rosbag recording.
        """
        if self.record_process is None:
            print("No recording process to stop.")
            return

        try:
            self.record_process.terminate()
            self.record_process.wait()
            print("Rosbag recording stopped.")
        except Exception as e:
            raise RuntimeError(f"Failed to stop recording: {e}")
        finally:
            self.record_process = None

    def drop_last_recorded_file(self):
        if self.record_process is not None:
            raise RuntimeError("Recording is already in progress. Call stop() before deleting a recording.")
        if not self._latest_filename:
            raise ValueError(f"self._latest_filename is None. You need to record a file before you can delete it.")
        latest_bag_name = f"{self._latest_filename}.bag"
        os.remove(latest_bag_name)
        print(f"Dropped {latest_bag_name}")
        self._latest_filename = None

    def commit_last_recorded_file(self):
        if self.record_process is not None:
            raise RuntimeError("Recording is already in progress. Call stop() before committing a recording.")
        if not self._latest_filename:
            raise ValueError(f"self._latest_filename has no value. You need to record a file before you can commit it.")
        latest_bag_name = f"{self._latest_filename}.bag"
        os.rename(src=latest_bag_name, dst=latest_bag_name.replace(".tmp", ""))

# Example usage:
if __name__ == "__main__":
    recorder = ROSBagRecorder()

    topics = ["/topic1", "/topic2"]
    filename = "example.bag"

    try:
        recorder.start(topics, filename)
        # Recording will run until stopped
        print("Press Ctrl+C to stop recording...")
        while True:
            pass  # Keep the program running until interrupted
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping recording...")
    finally:
        recorder.stop()
