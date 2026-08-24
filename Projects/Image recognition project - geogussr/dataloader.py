import msgpack
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import math
from geopy.geocoders import Nominatim

records = []
with open("data/shard_0.msg", "rb") as f:
    unpacker = msgpack.Unpacker(f, raw=False)
    for record in unpacker:
        records.append(record)

print(len(records))
for label, item in records[0].items():
    print(label, item)
imgs = [Image.open(BytesIO(records[i]["image"])) for i in range(len(records))]



n = min(12, len(imgs))  # display up to 12 images
cols = 4
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
axes = axes.flatten()

for i in range(n):
    axes[i].imshow(imgs[i])
    axes[i].axis("off")

# Hide unused subplot slots
for i in range(n, len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.show()

for label, item in records[3].items():
    if label == "latitude":
        latitude = item
    if label == "longitude":
        longitude = item
geolocator = Nominatim(user_agent="my_app")
location = geolocator.reverse(f"{latitude}, {longitude}")

print(location.address)