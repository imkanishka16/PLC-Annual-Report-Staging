import fitz  # PyMuPDF
import os

def split_pdf_by_pages(input_pdf, output_folder, page_ranges):
    """
    Splits a PDF into multiple parts based on the specified page ranges.

    :param input_pdf: Path to the input PDF file.
    :param output_folder: Directory to save the split PDFs.
    :param page_ranges: List of tuples, each representing (start_page, end_page).
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output directory if not exists
    doc = fitz.open(input_pdf)

    for i, (start, end) in enumerate(page_ranges):
        new_doc = fitz.open()  # Create a new empty PDF
        new_doc.insert_pdf(doc, from_page=start-1, to_page=end-1)  # PyMuPDF is 0-indexed

        output_path = os.path.join(output_folder, f"part_{i+1}_pages_{start}_to_{end}.pdf")
        new_doc.save(output_path)
        new_doc.close()
        print(f"Saved: {output_path}")

    doc.close()

# Example Usage
input_pdf = "annual_report_2024.pdf" 
output_folder = "pdf/"
page_ranges = [(1,53),(34,462)]  

split_pdf_by_pages(input_pdf, output_folder, page_ranges)