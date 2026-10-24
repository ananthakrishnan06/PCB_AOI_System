from datetime import datetime


def generate_report(result, image_name):

    report = []

    report.append("=" * 70)
    report.append("PCB AUTOMATIC OPTICAL INSPECTION REPORT")
    report.append("=" * 70)

    report.append("")
    report.append(f"Image      : {image_name}")
    report.append(f"Date & Time: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    report.append(f"Status     : {result['status']}")
    report.append(f"Total Defects : {result['total']}")
    report.append("")

    report.append("-" * 70)
    report.append(
        f"{'ID':<4}"
        f"{'Defect':<20}"
        f"{'Confidence':<15}"
        f"{'Center':<20}"
    )

    report.append("-" * 70)

    for i, defect in enumerate(result["defects"], start=1):

        report.append(
            f"{i:<4}"
            f"{defect['class']:<20}"
            f"{str(defect['confidence'])+'%':<15}"
            f"{str(defect['center']):<20}"
        )

    report.append("")
    report.append("Summary")
    report.append("-" * 30)

    for name, count in result["summary"].items():
        report.append(f"{name:<20}: {count}")

    report.append("")
    report.append("=" * 70)

    return "\n".join(report)