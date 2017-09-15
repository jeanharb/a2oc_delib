import seaborn
import matplotlib.pyplot as plt
import sys, time
import numpy as np

plt.ion()
fig = plt.figure()
plt.show(block=False)
refresh_rate = 5.0

def handle_close(evt):
  sys.exit()

show_term = "--term" in sys.argv
while True:
  try:
    data = []
    indices = []
    plt.clf()
    weight_moves = False
    for i in range(len(sys.argv[1:])):
      if "--" in sys.argv[i+1]:
        continue
      d = []
      e = []
      filename = sys.argv[i+1]
      if ".csv" not in filename: filename += "/data.csv"
      f = open(filename, "rb")
      for j, line in enumerate(f):
        if not line.split(",")[0].isdigit(): continue
        if weight_moves or ("," in line):
          d.append(float(line.split(",")[1+show_term]))
          e.append(int(line.split(",")[0]))
          weight_moves = True
        else:
          d.append(int(line))
          weight_moves = False
      f.close()
      data.append(d)
      indices.append(e)

    a = int(max([len(each) for each in data])/250)+1
    #weight_moves = False
    if weight_moves:
      a = int(float(max([i[-1] for i in indices]))/250)
    d2 = []
    all_p = []
    i = -1
    for temp_i in range(len(sys.argv[1:])):
      if "--" in sys.argv[temp_i+1]:
        continue
      i += 1
      if weight_moves:
        frame_interval = a
        new_matrix = []
        one_row = []
        counter = 0
        count = 0
        while count < len(data[i]):
          if indices[i][count] > (counter+1)*frame_interval:
            if len(one_row) == 0:
              if len(new_matrix) == 0:
                one_row = [data[i][count]]
              else:
                one_row = [new_matrix[-1]]
            new_matrix.append(np.mean(one_row))
            one_row = []
            counter += 1
          else:
            one_row.append(data[i][count])
            count += 1
        p, = plt.plot(np.array(range(len(new_matrix)))*frame_interval, np.array(new_matrix))
      else:
        p, = plt.plot(np.array(data[i][:-(len(data[i])%a)]).reshape(((len(data[i])-(len(data[i])%a))/a,a)).mean(axis=1).flatten())
      all_p.append(p)
    legends = []
    for dd in sys.argv[1:]:
      if "--" not in dd: legends.append(dd.split("/")[-2])
    plt.legend(all_p, legends, loc=2)
    fig.canvas.mpl_connect('close_event', handle_close)
    plt.draw()
    plt.pause(refresh_rate)
  except Exception, e:
    print e
    time.sleep(2)
    pass
