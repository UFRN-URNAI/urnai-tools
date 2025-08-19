class ClipRangeExponentialSchedule:
    def __init__(self, initial_value, final_value, total_iterations):
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_iterations = total_iterations
        self.current_iteration = 0

    def __call__(self, unused_progress_remaining : float) -> float:
        base = self.final_value + (1 - self.initial_value)

        global_progress_remaining = self.current_iteration/self.total_iterations

        if self.current_iteration <= self.total_iterations:
            new_value = base ** global_progress_remaining - (1 - self.initial_value)
        else:
            global_progress_remaining = 1
            new_value = self.final_value

        print("\nSCHEDULE UPDATING VALUE TO: ", new_value,
            " | ", global_progress_remaining * 100, "% TO FINAL VALUE\n")
        
        self.current_iteration += 1

        return new_value