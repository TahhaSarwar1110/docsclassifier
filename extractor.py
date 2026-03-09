import re

def extract_invoice(text):
    return {
        "invoice_number": re.search(r"Invoice\s*#?\s*(\S+)", text),
        "date": re.search(r"Date:\s*(.*)", text),
        "company": re.search(r"Company:\s*(.*)", text),
        "total_amount": re.search(r"\$([\d\.]+)", text)
    }

def extract_resume(text):
    lines = text.split("\n")
    name = lines[0] if lines else None
    email = re.search(r"\S+@\S+", text)
    phone = re.search(r"\+?\d[\d\- ]{7,}", text)
    exp = re.search(r"(\d+)\s+years", text)

    return {
        "name": name,
        "email": email.group(0) if email else None,
        "phone": phone.group(0) if phone else None,
        "experience_years": int(exp.group(1)) if exp else None
    }

def extract_utility(text):
    return {
        "account_number": re.search(r"Account.*:\s*(\S+)", text),
        "date": re.search(r"Date:\s*(.*)", text),
        "usage_kwh": re.search(r"(\d+)\s*kWh", text),
        "amount_due": re.search(r"\$([\d\.]+)", text)
    }

def run_extraction(doc_class, text):
    if doc_class == "Invoice":
        return {k: v.group(1) if v else None for k, v in extract_invoice(text).items()}
    if doc_class == "Resume":
        return extract_resume(text)
    if doc_class == "Utility Bill":
        return {k: v.group(1) if v else None for k, v in extract_utility(text).items()}
    return {}
