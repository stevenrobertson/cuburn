import sys, json
sys.path.insert(0, '.')
from cuburn.genome import Genome, json_encode_genome

with open(sys.argv[1]) as fp:
    gnm = Genome(json.load(fp))

gnm['xforms'].pop('final', False)
n = len(gnm['xforms'])
for i, v in enumerate(gnm['xforms'].values()):
    v['variations'] = {'linear':{'weight':1}}
    o = 0.002 * i / n

    csp, cv = v['color_speed'], v['color']
    v['affine'] = {
            'angle': 45,
            'spread': 45,
            'magnitude': {'x': 0.001, 'y': [0, 1-csp(0), 1, 1-csp(1)]},
            'offset': {'x': [0, o - 8./9, 1, o + 8./9],
                       'y': [0, cv(0)*csp(0), 1, cv(1)*csp(1)]}
        }
    v.pop('post', False)
gnm['color']['highlight_power'] = 0
gnm['camera'].update(center={'x': 0,'y':-0.5}, rotation=0, scale=9. / 16,
        filter_width=2)
name = sys.argv[1].rsplit('.', 1)[0] + '.showpal.json'
with open(name, 'w') as fp:
    fp.write(json_encode_genome(gnm))
print 'okay, now run:'
print 'python main.py --duration 0.5 --fps 1 -p 720p %s' % name
