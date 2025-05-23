import numpy as np

def _add_blocks(model, blocks):
    for i, block in enumerate(blocks):
        model.worldbody.add('geom', name=f'block_{i+1}', conaffinity="1", contype="0", type='box', pos=block['pos'], euler=block.get("euler", "0 0 0"), rgba=block['rgba'], size=block['size'])
    return model.to_xml_string()

def has_contact(physics):
    for a, b in zip(physics.data.contact.geom1, physics.data.contact.geom2):
       if ("block" in physics.model.id2name(a, 'geom')) or ("block" in physics.model.id2name(b, 'geom')):
          return 0
       

min_pos_x, max_pos_x = -10, 10
min_pos_y, max_pos_y = -10, 10

min_x_sz, max_x_sz = .1, .5
min_y_sz, max_y_sz = .1, .5
min_height, max_height = .1, .5

min_angle, max_angle = 0, 90

RGBA_COLOR = "0.9 0.4 0.6 1"

def _block_generator(n):
    blocks = []
    for i in range(n):
        pos_x, pos_y = np.random.uniform(low=min_pos_x, high=max_pos_x), np.random.uniform(low=min_pos_y, high=max_pos_y)
        l, w = np.random.uniform(low=min_x_sz, high=max_x_sz), np.random.uniform(low=min_x_sz, high=max_x_sz)
        height = np.random.uniform(low=min_height, high=max_height)
        pos_z = height
        angle = np.random.randint(low=min_angle, high=max_angle)
        blocks += [{"pos": f"{pos_x:.2f} {pos_y:.2f} {pos_z:.2f}", "size": f"{l:.2f} {w:.2f} {height:.2f}", "rgba": RGBA_COLOR, "euler": f"0 0 {angle}"}]
    return blocks