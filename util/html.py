import dominate
from dominate.tags import *
from dominate.util import raw
import os



refresh_js = '''
function refresh_single(refresh_id) {
    var source = document.getElementById(refresh_id).src;
    var timestamp = (new Date()).getTime();
    var newUrl = source + '?_=' + timestamp;
    document.getElementById(refresh_id).src = newUrl;
}

function refresh(refresh_ids) {
    var arrayLength = refresh_ids.length;
    for (var i = 0; i < arrayLength; i++) {
        console.log(refresh_ids[i]);
        refresh_single(refresh_ids[i]);
    }
}
'''


class HTML:
    def __init__(self, web_dir, title, reflesh=0):
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        # print(self.img_dir)

        self.doc = dominate.document(title=title)

        with self.doc.head:
            script(raw(refresh_js))

        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(reflesh))

        with self.doc:
            h1(title, style="font-size: x-large;")

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str, style="margin: 0; font-size: medium;")

    def add_text(self, str):
        with self.doc:
            p(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed; border: 0px solid black;")
        self.doc.add(self.t)


    def add_media(self, ims, txts, links, type, width=400, title=None):
        # add refresh button
        with self.doc:
            with p(style="margin: 2px 0 0 0;"):
                if title is not None:
                    span(title, style="margin: 0; font-size: medium;")
                button("refresh", onclick="refresh(" + str(ims) + ")")
        self.add_table()
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word; border: 0px solid black;", halign="center", valign="top"):
                        with a(href=os.path.join('images', link)):
                            if type == 'image':
                                self.add_image(im, width)
                            elif type == 'video':
                                self.add_video(im, width)
                            else:
                                raise ValueError()
                        p(txt, style="margin: 0;")

    
    def add_image(self, im, width):
        img(style="width:%dpx" % width,
            src=os.path.join('images', im),
            id=im)
    
    def add_video(self, im, width):
        with video(style="width:%dpx" % width, id=im, autoplay="", loop="", muted="", inline="", playsinline=""):
            source(src=os.path.join('images', im), type="video/mp4")

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    links = []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
