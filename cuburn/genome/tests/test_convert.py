import unittest

import binascii
import numpy as np

from cuburn.genome import convert

def _make_palette_src():
    values = np.zeros((256,4), 'u1')
    values[:,0] = range(256)
    values[:,1] = 1
    values[:,2] = 2
    values[:,3] = 3
    # leave a newline in to make sure those get stripped
    return """<palettes><palette number="0" name="synthetic" data="%s
"/></palettes>""" % binascii.b2a_hex(values.tostring())

def _make_genome_src(palette=False):
    src = """
<flame time="0" size="1280 960" center="0.01 0.02" scale="40" oversample="2"
    filter="1" quality="500" batches="50" brightness="4" gamma="4"
    url="test.com" nick="strobe" {paletteidx}>
    {color}
    <xform weight="0.1" color="0" hyperbolic="0.1"
        coefs="01 0.2 -0.3 0.4 -0.5 0.6"/>
</flame>"""
    args = {'paletteidx': '', 'color': ''}
    if palette:
        args['paletteidx'] = 'palette="0"'
    else:
        args['color'] = '<color index="0" rgb="1 2 3"/>'
    return src.format(**args)

class XMLPaletteParserTest(unittest.TestCase):
    def test_parse(self):
        parser = convert.XMLPaletteParser(_make_palette_src())
        self.assertIn('synthetic', parser.names)
        self.assertIn(0, parser.numbers)
        self.assertEquals([0,1/255.,2/255.,3/255.], list(parser.numbers[0][0]))
        self.assertEquals([1,1/255.,2/255.,3/255.], list(parser.numbers[0][255]))

class ConversionTest(unittest.TestCase):
    def test_parse(self):
        parsed = convert.XMLGenomeParser.parse(_make_genome_src())
        converted = convert.flam3_to_node(parsed[0])
        palette = converted.pop('palette')
        self.maxDiff = None
        self.assertEquals(dict(
            type='node',
            author=dict(url='http://test.com', name='strobe'),
            camera=dict(dither_width=1.0, scale=0.03125,
                        center=dict(x=0.01, y=0.02)),
            filters=dict(
                logscale=dict(brightness=4.0),
                colorclip=dict(gamma=4.0)),
            xforms={
                '0': dict(
                    color=0.0,
                    variations=dict(hyperbolic=dict(weight=0.1)),
                    pre_affine=dict(
                        spread=32.220017414088105,
                        angle=[20.91008494006789, -360],
                        magnitude=dict(x=1.019803902718557, y=0.5),
                        offset=dict(x=-0.5, y=-0.6)),
                    weight=0.1)
            }), converted)
        self.assertEquals('rgb8', palette[0])
        self.assertEquals('AQID////', palette[1][:8])

    def test_parse_stock_palette(self):
        try:
            convert.XMLPaletteParser.lookup(0)
        except:
            # No system palettes installed, just skip the test
            raise
        parsed = convert.XMLGenomeParser.parse(_make_genome_src(True))
        converted = convert.flam3_to_node(parsed[0])
        palette = converted['palette']
        self.assertEquals('rgb8', palette[0])
        self.assertEquals('ALnqAMHu', palette[1][:8])

