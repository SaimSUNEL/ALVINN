import matplotlib.pyplot as plt


road_name = "Road4"
error_graph_file_name = "MlpRegressionRGB3Images"+road_name+"_graph.dat"

error_graph_file = open(error_graph_file_name, "r")


iteration_axis = []
error_axis = []
validation_error_axis = []


while True:
    error = error_graph_file.readline()
    if error == "":
        break
    values = error.split(" ")
    iteration_axis.append(int(values[0]))
    # Train error
    error_axis.append(float(values[1]))
    validation_error_axis.append(float(values[2]))

error_iteration_number = max(iteration_axis) + 1
error_graph_file.close()
error_graph_file = open(error_graph_file_name, "a")
plt.subplot(211)
plt.plot(iteration_axis, error_axis, label="Error")
plt.subplot(212)

plt.plot(iteration_axis, validation_error_axis, label="Error")
plt.show()