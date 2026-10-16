class ChasmError(Exception):
    """Base class for user-actionable CHASM errors."""


class DocumentError(ChasmError):
    pass


class UnsupportedDocumentError(DocumentError):
    pass


class ExtractionError(DocumentError):
    pass

