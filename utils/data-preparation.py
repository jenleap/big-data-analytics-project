import matplotlib.pyplot as plt


def build_hist_distribution(col, x_label, y_label):
    plt.figure(figsize=(8, 5))
    plt.hist(col, bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {x_label.capitalize()}s per {x_label.capitalize()}")
    plt.xlabel(f"Number of {x_label.capitalize()}s")
    plt.ylabel(f"Number of {y_label.capitalize()}s")
    plt.tight_layout()
    plt.savefig(f"../graphs/{y_label}_{x_label}_counts_hist.png") 
    plt.close()

def build_bar_chart(col, x_label, y_label):
    plt.figure(figsize=(10, 6))
    plt.bar(col.index, col.values, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {x_label.capitalize()}s per {y_label.capitalize()}")
    plt.xlabel(f"Number of {x_label.capitalize()}s")
    plt.ylabel(f"Number of {y_label.capitalize()}s")
    plt.tight_layout()
    plt.savefig(f"../graphs/{y_label}_{x_label}_distribution.png")
    plt.close()