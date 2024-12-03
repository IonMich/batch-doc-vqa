import random
import string
import os

import fitz


def pdf_to_imgs(filepath, pages_i=-1, dpi=150, output_dir="."):
    """Converts a pdf file to images."""
    doc = fitz.open(filepath)

    n_pages = doc.page_count
    if pages_i == -1:
        num_pages_per_doc = n_pages
    else:
        num_pages_per_doc = pages_i
    if n_pages % pages_i != 0:
        raise ValueError(
            "The number of pages in the pdf is not a multiple of num_pages_per_doc."
        )

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    n_docs = n_pages // num_pages_per_doc
    print(f"File metadata: {doc.metadata}")
    # store also a csv with doc, page, and filename
    doc_info = []
    for i in range(n_docs):
        print(f"Document {i+1}/{n_docs}", end="\r")

        # we want to avoid name collisions, so we generate a random string
        # to append to the filename
        random_string = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=8)
        )

        for j in range(num_pages_per_doc):
            page = doc.load_page(i * num_pages_per_doc + j)
            img_filename = f"doc-{i}-page-{j+1}-{random_string}.png"
            img_filepath = os.path.join(output_dir, img_filename)
            pix = page.get_pixmap(dpi=dpi)
            pix_bytes = pix.tobytes()
            with open(img_filepath, "wb") as f:
                f.write(pix_bytes)
            doc_info.append((i, j + 1, img_filename))

    doc.close()
    with open(os.path.join(output_dir, "doc_info.csv"), "w") as f:
        f.write("doc,page,filename\n")
        for doc, page, filename in doc_info:
            f.write(f"{doc},{page},{filename}\n")
    return doc_info


if __name__ == "__main__":
    # example usage:
    # python pdf_to_imgs.py --filepath "path/to/pdf" --pages_i 2 --dpi 150 --output_dir "path/to/output"
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str)
    parser.add_argument(
        "--pages_i",
        type=int,
        default=-1,
        help=(
            "Number of pages per individual document, assuming the --filepath pdf is a concatenation of multiple documents. "
            "Defaults to -1, which means that the entire pdf is considered a single document."
        ),
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    pdf_to_imgs(args.filepath, args.pages_i, args.dpi, args.output_dir)
