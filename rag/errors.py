class UpstreamFailure(RuntimeError):
    def __init__(self, stage: str, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.stage = stage
        self.status = status
