"""Serve the companion site locally.

    python web/serve.py            # http://localhost:8000/

The site is plain ES modules, so it needs a server only because module
imports and fetch() do not work from file:// URLs -- there is no build
step to run and nothing to watch. Edit a file and reload.

The one piece of routing: pages ask for `data/polyhedra/...` relative to
web/, and the database actually lives at the repository's
`data/polyhedra/`. Rather than copy 12 MB into web/ to make the paths
line up, this server maps that prefix straight onto the real directory.
The Pages workflow copies instead, because a static host cannot route --
either way the URL a page requests is the same in development and in
production, which is the property that matters.
"""
import http.server
import os
import sys

WEB = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(WEB)
DB_PREFIX = "/data/polyhedra/"
DB_ROOT = os.path.join(PROJ, "data", "polyhedra")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=WEB, **kw)

    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith(DB_PREFIX):
            rel = clean[len(DB_PREFIX):]
            # Refuse to walk out of the database directory.
            target = os.path.normpath(os.path.join(DB_ROOT, rel))
            if os.path.commonpath([target, DB_ROOT]) == DB_ROOT:
                return target
        return super().translate_path(path)

    def end_headers(self):
        # The database and the thumbnails change only when regenerated,
        # but during development they change under a reload, so don't
        # let the browser hold a stale copy.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "404" in (fmt % args):
            super().log_message(fmt, *args)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    if not os.path.isdir(DB_ROOT):
        print("warning: %s not found -- the catalogue will be empty" % DB_ROOT)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("Math Art companion site: http://localhost:%d/" % port)
    print("  polyhedra module:      http://localhost:%d/modules/polyhedra.html"
          % port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
