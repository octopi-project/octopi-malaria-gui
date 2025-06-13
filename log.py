def report(timing_data, fov_id,logger):
    processes = list(timing_data.keys())
    max_end_time = max(timing_data[process]['end'] for process in processes)
    min_start_time = min(timing_data[process]['start'] for process in processes)
    total_duration = max_end_time - min_start_time

    # Calculate the maximum process name length for alignment
    max_name_length = max(len(process) for process in processes)

    for process in processes:
        start_time = timing_data[process]['start'] - min_start_time
        end_time = timing_data[process]['end'] - min_start_time
        duration = end_time - start_time

        # Calculate the position and width of the bar
        bar_start = int((start_time / total_duration) * 50)
        bar_width = max(1, int((duration / total_duration) * 50))

        # Create the progress bar
        progress_bar = ' ' * bar_start + '=' * bar_width + ' ' * (50 - bar_start - bar_width)

        logger.info(f"{process.ljust(max_name_length)} | {duration:.3f}s | {progress_bar} |")

    logger.info(f"Total time: {total_duration:.3f}s")
    logger.info(f"{'=' * 50}")
