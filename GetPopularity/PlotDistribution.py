import matplotlib.pyplot as plt
import seaborn as sns
import json
import matplotlib.ticker as ticker
import numpy as np

with open('/Users/graceli/Desktop/ECE324/Project/GetPopularity/DataDictFullwithPop.json') as json_file:
    data = json.load(json_file)   

count = {}
group1 = 0
for music in data:
    pop = data[music]["Popularity"]
    if pop not in count:
        count[pop] = 1
    count[pop] += 1

# for popularity in count:
#     if popularity >= 47:
#         group1 = group1 + popularity
        
# count.pop(0)
count[0] = 400
# sortedcount = sorted(count)
sortedcount = sorted(count.items(), key=lambda x:x[0])
print(sortedcount)
convert_count = dict(sortedcount)

names = list(convert_count.keys())
values = list(convert_count.values())

ax = plt.bar(range(len(convert_count)), values, tick_label=names)
ax2 = plt.bar(range(len(convert_count)), values, tick_label=names)
# ax.set_ylim(0, 400)  # outliers only
# ax2.set_ylim(7500, 8100)  # most of the data


plt.hist(names,bins=300)
plt.xticks(np.arange(0, 65, 5.0))
plt.show()


# # Now let's make two outlier points which are far away from everything.
# pts[[3, 14]] += .8

# # If we were to simply plot pts, we'd lose most of the interesting
# # details due to the outliers. So let's 'break' or 'cut-out' the y-axis
# # into two portions - use the top (ax) for the outliers, and the bottom
# # (ax2) for the details of the majority of our data
# f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

# # plot the same data on both axes
# ax.plot(pts)
# ax2.plot(pts)

# # zoom-in / limit the view to different portions of the data
# ax.set_ylim(.78, 1.)  # outliers only
# ax2.set_ylim(0, .22)  # most of the data

# hide the spines between ax and ax2
# ax.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax.xaxis.tick_top()
# ax.tick_params(labeltop=False)  # don't put tick labels at the top
# ax2.xaxis.tick_bottom()

# This looks pretty good, and was fairly painless, but you can get that
# cut-out diagonal lines look with just a bit more work. The important
# thing to know here is that in axes coordinates, which are always
# between 0-1, spine endpoints are at these locations (0,0), (0,1),
# (1,0), and (1,1).  Thus, we just need to put the diagonals in the
# appropriate corners of each of our axes, and so long as we use the
# right transform and disable clipping.

# d = .015  # how big to make the diagonal lines in axes coordinates
# # arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
# ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

# kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'

plt.show()