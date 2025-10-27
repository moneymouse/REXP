from tqdm import tqdm, trange

class MetricDisplay:
    label: str
    value: float

class EpochWrapper:
    def __init__(self, epochs, position=0, leave=True):
        self.epochs = epochs
        self.position = position
        self.leave = leave

    def __call__(self, func: callable[..., list[MetricDisplay]]):
        def wrapper(*args, **kwargs):
            with trange(total=self.epochs, position=self.position, leave=self.leave) as pbar:
                print()
                for epoch in range(self.epochs):
                    pbar.set_description(f"Epoch{epoch}")
                    metrics = func(*args, **kwargs)
                    if(epoch % 10 == 0):
                        tqdm.write("\033[F \033[K \r", end="")
                        for metric in metrics:
                            tqdm.write(f"{metric.label}: {metric.value:.4f}")
                    pbar.update()
        return wrapper