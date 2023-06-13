from datetime import datetime, timezone


class ZoomTime():
    """ A ZoomTime converts times returned by Zoom API into
    objects that can be compared.

    """

    def __init__(self, time):
        self.time = datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')

    def utc_to_local(self):
        return self.time.replace(tzinfo=timezone.utc).astimezone(tz=None)

