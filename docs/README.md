# Sensori website

This folder contains the public Sensori project site. It uses Jekyll, SCSS,
vanilla JavaScript and Markdown; there is no JavaScript framework or database.

## Preview locally

```bash
/opt/homebrew/opt/ruby/bin/bundle config set --local path vendor/bundle
/opt/homebrew/opt/ruby/bin/bundle install
/opt/homebrew/opt/ruby/bin/bundle exec jekyll serve --livereload
```

Open `http://127.0.0.1:4000/Sensori/`. Jekyll rebuilds the site when a source file
changes. These commands use Homebrew Ruby because the macOS system Ruby is too
old for this build. On Linux, or when a current Ruby is already first on your
path, the shorter `bundle ...` form is sufficient.

## Edit the site

- Homepage: `index.html`
- Research case studies: `_research/*.md`
- White paper: `white-paper.md`
- Tutorials: `_tutorials/*.md`
- Navigation: `_data/navigation.yml`
- Colours and typography: `_sass/_tokens.scss`
- Layout and component styles: `_sass/_components.scss`

Set the final paper, code and model URLs in `_config.yml`. Until those values
are present, the site deliberately shows “Available at release”.

## Publish with GitHub Pages

In the GitHub repository, open **Settings → Pages**. Under **Build and
deployment**, select **Deploy from a branch**, then choose the `main` branch and
the `/docs` folder. Changes to `docs/` on `main` will then trigger a Pages build.

If the site moves to a custom domain or a dedicated repository, update `url`
and `baseurl` in `_config.yml`.
