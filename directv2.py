import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist

# Read the output files
data_plus1 = pd.read_csv(r"C:\Users\kamil\OneDrive - Lancaster University\ANU\PHYS3042\Bubles\W_P740_NA_w_gold_repeat annealing\W_He_740C_BF_ZL_400kx_Defoc_+1mu_After_cool_crop1.csv")  # Adjust file name and path as needed
data_minus1 = pd.read_csv(r"C:\Users\kamil\OneDrive - Lancaster University\ANU\PHYS3042\Bubles\W_P740_NA_w_gold_repeat annealing\W_He_740C_BF_ZL_400kx_Defoc_-1mu_After_cool_crop.csv")  # Adjust file name and path as needed
#data_plus1 = pd.read_csv(r"C:\Users\kamil\OneDrive - Lancaster University\ANU\PHYS3042\Bubles\Useful files\RT\new fr\W_He_740C_BF_ZL_400kx_Defoc_+1mu_RT_crop.csv")
#data_minus1 = pd.read_csv(r"C:\Users\kamil\OneDrive - Lancaster University\ANU\PHYS3042\Bubles\Useful files\RT\new fr\W_He_740C_BF_ZL_400kx_Defoc_-1mu_RT_cropa.csv")
# Print the first few rows of data for inspection
print(data_plus1.head())
print(data_minus1.head())


# Select points (adjust indices based on your actual data inspection)

#Aftercool

v1_o = np.array([463, 298])
v2_o = np.array([562, 660])
v1_u = np.array([402, 299])
v2_u = np.array([506, 660])

#RT
'''
v1_o = np.array([430, 250])
v2_o = np.array([537, 596])#v2_o = np.array([520, 600])
v1_u = np.array([366, 236])
v2_u = np.array([468, 588])
'''
pivot = v1_o + (v2_o - v1_o) / 2

# Plotting initial positions
plt.figure(figsize=(12, 10))

plt.scatter(data_plus1['x'], data_plus1['y'], color='blue', label='Over-focused')
plt.scatter(data_minus1['x'], data_minus1['y'], color='green', label='Under-focused')
# Draw vectors for original datasets
plt.arrow(v1_o[0], v1_o[1], v2_o[0] - v1_o[0], v2_o[1] - v1_o[1], head_width=3, head_length=1, fc='blue', ec='blue')
plt.arrow(v1_u[0], v1_u[1], v2_u[0] - v1_u[0], v2_u[1] - v1_u[1], head_width=3, head_length=1, fc='red', ec='green')

plt.scatter(pivot[0], pivot[1], color='magenta', s=300, marker='x', label='Pivot Point', linewidths= 3)


plt.xlabel('x Position',fontsize = 35)
plt.ylabel('y Position',fontsize = 35)
plt.title('Initial positions of blobs with transformation vectors \nfor the Aftercool sample',fontsize = 30)
plt.legend(fontsize = 20)
plt.grid(True)
plt.xticks(fontsize = 35)
plt.yticks(fontsize = 35)

plt.gca().invert_yaxis()

plt.show()


#******

#******

# Define transformation functions
def rotate_points(points, angle, origin):
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_points = np.dot(points - origin, R.T) + origin
    return rotated_points

def transform_points(data, translation, scale_factor, angle, pivot):
    points = data[:, :2]  # Extract x, y coordinates
    transformed_points = (points + translation) * scale_factor
    transformed_points = rotate_points(transformed_points, angle, pivot)
    # Include the radius data unchanged in the transformation result
    return np.hstack((transformed_points, data[:, 2:3]))

# Calculate transformation parameters
translation = v1_o - v1_u  # Translation for the bottom point of the vectors
distance_o = np.linalg.norm(v2_o - v1_o)
distance_u = np.linalg.norm(v2_u - v1_u)
scale_factor = distance_o / distance_u
angle = np.arctan2(v2_o[1] - v1_o[1], v2_o[0] - v1_o[0]) - np.arctan2(v2_u[1] - v1_u[1], v2_u[0] - v1_u[0])

# Print the shift, scale, and rotation
print("Translation vector:", translation)
print("Scaling factor:", scale_factor)
print("Rotation angle (degrees):", angle*180/np.pi)




# Transform all under-focused points (include radii in the transformation)
data_minus1_with_radii = np.hstack((data_minus1[['x', 'y']].to_numpy(), data_minus1[['radius']].to_numpy().reshape(-1, 1)))
data_minus1_transformed = transform_points(data_minus1_with_radii, translation, scale_factor, angle, pivot)

# Define a threshold for overlapping
overlap_threshold = 8  # Adjust this value based on your data scale

# Highlight specific points - Triplet seen in tif file

#Aftercool
highlight_points = np.array([[472, 426], [491, 426], [479, 443]])
#RT
#highlight_points = np.array([[437, 377], [456, 378], [445, 397],[533,607],[467,427]])
#highlight_points = np.array([[388, 428], [404, 251], [431, 218],[469,513]])


# Original Plot of Unfiltered Data
plt.figure(figsize=(12, 10))
plt.scatter(data_plus1['x'], data_plus1['y'], color='blue', s=data_plus1['radius']*100, label='Over-focused (original)')
plt.scatter(data_minus1_transformed[:, 0], data_minus1_transformed[:, 1], color='green', s=data_minus1_transformed[:, 2]*100, label='Under-focused transformed')

# Filtered Data Plot
#plt.scatter(filtered_data_plus1['x'], filtered_data_plus1['y'], color='blue', s=filtered_data_plus1['radius']*100, label='Over-focused (filtered)')
#plt.scatter(filtered_data_minus1_transformed[:, 0], filtered_data_minus1_transformed[:, 1], color='green', s=filtered_data_minus1_transformed[:, 2]*100, label='Under-focused transformed (filtered)',alpha=0.5)

# Highlighted points from the original specification (if still relevant)
# Plotting points of interest with only one label
for i, point in enumerate(highlight_points):
    if i == 0:
        plt.scatter(point[0], point[1], s=300, color='yellow', edgecolors='black', marker='D', alpha=0.5, label='Point of interest')
    else:
        plt.scatter(point[0], point[1], s=300, color='yellow', edgecolors='black', marker='D', alpha=0.5)


params_text = f"Translation:{translation}\nScale Factor: {scale_factor:.4f}\nRotation: {angle*180/np.pi:.3f}°"
plt.text(0.05, 0.3, params_text, transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))

plt.xlabel('x Position',fontsize = 35)
plt.ylabel('y Position',fontsize = 35)
plt.title('Under-focused points mapped onto \nthe over-focused points',fontsize = 35)
plt.legend(fontsize = 20)
plt.grid(True)
plt.xticks(fontsize = 35)
plt.yticks(fontsize = 35)
plt.gca().invert_yaxis()

#plt.show()




####
# Assuming data_plus1 and data_minus1_transformed are already defined
# Calculate the distances between all points in data_plus1 and data_minus1_transformed
distances = cdist(data_plus1[['x', 'y']], data_minus1_transformed[:, :2])

# Create a mask for distances less than the overlap threshold
#overlap_threshold = 8 # Define your own threshold
min_distances = np.min(distances, axis=1)
min_indices = np.argmin(distances, axis=1)
mask_plus1 = min_distances <= overlap_threshold

# Filter data_plus1 based on the mask
filtered_data_plus1 = data_plus1[mask_plus1]

# Extract the corresponding closest points from data_minus1_transformed
# Using unique indices to ensure one-to-one mapping
unique_indices, indices_position = np.unique(min_indices[mask_plus1], return_index=True)
filtered_data_minus1_transformed = data_minus1_transformed[unique_indices]

# For data_minus1_transformed to data_plus1
distances_T = distances.T
min_distances_T = np.min(distances_T, axis=1)
min_indices_T = np.argmin(distances_T, axis=1)
mask_minus1_transformed = min_distances_T <= overlap_threshold

# Filter data_minus1_transformed based on the mask
filtered_data_minus1_transformed = data_minus1_transformed[mask_minus1_transformed]

# Extract corresponding closest points from data_plus1
unique_indices_T, indices_position_T = np.unique(min_indices_T[mask_minus1_transformed], return_index=True)
filtered_data_plus1 = data_plus1.iloc[unique_indices_T]

# Plotting to visualize the results
plt.figure(figsize=(12,10))
plt.scatter(data_plus1['x'], data_plus1['y'],s=data_plus1['radius']*100 ,color='lightblue', alpha=0.7, label='+1 (original)')
plt.scatter(data_minus1_transformed[:, 0], data_minus1_transformed[:, 1], s=data_minus1_transformed[:, 2]*100, color='lightgreen', alpha=0.7, label='-1 transformed')
plt.scatter(filtered_data_plus1['x'], filtered_data_plus1['y'], s=filtered_data_plus1['radius']*100,color='blue', label='+1 (filtered)')
plt.scatter(filtered_data_minus1_transformed[:, 0], filtered_data_minus1_transformed[:, 1], s=filtered_data_minus1_transformed[:, 2]*100,color='green', label='-1 transformed (filtered)')

for i, point in enumerate(highlight_points):
    if i == 0:
        plt.scatter(point[0], point[1], s=300, color='yellow', edgecolors='black', marker='D', alpha=0.5, label='Point of interest')
    else:
        plt.scatter(point[0], point[1], s=300, color='yellow', edgecolors='black', marker='D', alpha=0.5)


plt.xlabel('x Position', fontsize = 35)
plt.ylabel('y Position',fontsize = 35)
plt.title('Mapped and Filtered Bubble Positions\n Overlap threshold {}'.format(overlap_threshold),fontsize = 35)
plt.legend(loc = 'best',fontsize = 18)
plt.grid(True)
plt.xticks(fontsize = 35)
plt.yticks(fontsize = 35)
plt.gca().invert_yaxis()

plt.savefig(r'C:\Users\kamil\OneDrive - Lancaster University\ANU\PHYS3042\Bubles\Useful files\Direct results\Filtered_Overlap_{}_Aftercool.png'.format(overlap_threshold), dpi=300)
plt.show()


# Create the figure with adjusted aspect ratio
fig, ax = plt.subplots(figsize=(12, 10))  

fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

# Plot the data
ax.scatter(data_plus1['x'],data_plus1['y'], s=data_plus1['radius'], color='lightblue', alpha=0, label='Over-focused (original)')
ax.scatter(data_minus1_transformed[:, 0], data_minus1_transformed[:, 1], s=data_minus1_transformed[:, 2], color='lightgreen', alpha=0, label='Under-focused transformed (original)')
ax.scatter(filtered_data_plus1['x'], filtered_data_plus1['y'], s=filtered_data_plus1['radius'], color='blue', label='Over-focused (filtered)')
ax.scatter(filtered_data_minus1_transformed[:, 0], filtered_data_minus1_transformed[:, 1], s=filtered_data_minus1_transformed[:, 2], color='green', label='Under-focused transformed (filtered)')
# Highlighted points from the original specification (if still relevant)
for point in highlight_points:
    plt.scatter(point[0], point[1], s=200, color='yellow', edgecolors='black', marker='o', label='Highlighted Points', alpha=0.3)

# Invert x-axis if necessary
ax.invert_yaxis()

# Remove axes
ax.axis('off')

filtered_data_plus1 = pd.DataFrame(filtered_data_plus1)
filtered_data_minus1_transformed = pd.DataFrame(filtered_data_minus1_transformed)
# Save the plot with a transparent background
#plt.savefig(r'C:\Users\kamil\OneDrive - Lancaster University\ANU\PHYS3042\Bubles\Useful files\Direct results\Transparent_Overlap_{}_Aftercool.png'.format(overlap_threshold), transparent=True, dpi=300)
#filtered_data_plus1.to_csv(r"C:\Users\kamil\OneDrive - Lancaster University\ANU\PHYS3042\Bubles\Useful files\plus1_raddi_RT.csv", index=True)  # Adjust file name and path as needed
#filtered_data_minus1_transformed.to_csv(r"C:\Users\kamil\OneDrive - Lancaster University\ANU\PHYS3042\Bubles\Useful files\minus1_radii_RT.csv", index=True)  # Adjust file name and path as needed

#plt.show()


