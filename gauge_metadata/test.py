from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
)

result = ocr.ocr("pressureGauge_419_jpg.rf.gCHYGGvW7aTu7Qwxp7aC.jpg")

print(type(result))
print(result)