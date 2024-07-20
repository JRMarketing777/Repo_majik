To install an easy Markdown reader in the Linux shell, you can choose from a couple of lightweight options. Below are instructions for installing **mdless** and **Glow**, both of which are simple and effective Markdown viewers.

### Installing mdless

**mdless** is a Ruby gem that allows you to view Markdown files in the terminal with formatting.

1. **Install Ruby (if not already installed):**
   ```bash
   sudo apt update
   sudo apt install ruby
   ```

2. **Install mdless:**
   ```bash
   sudo gem install mdless
   ```

3. **Usage:**
   To view a Markdown file, simply run:
   ```bash
   mdless your_file.md
   ```
   Replace `your_file.md` with the name of your Markdown file.

### Installing Glow

**Glow** is a CLI tool that renders Markdown files beautifully in the terminal.

1. **For Ubuntu/Debian-based systems:**
   ```bash
   sudo mkdir -p /etc/apt/keyrings
   curl -fsSL https://repo.charm.sh/apt/gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/charm.gpg
   echo "deb [signed-by=/etc/apt/keyrings/charm.gpg] https://repo.charm.sh/apt/ * *" | sudo tee /etc/apt/sources.list.d/charm.list
   sudo apt update && sudo apt install glow
   ```

2. **For Arch-based systems:**
   ```bash
   sudo pacman -S glow
   ```

3. **For Fedora/RHEL-based systems:**
   ```bash
   echo '[charm]
   name=Charm
   baseurl=https://repo.charm.sh/yum/
   enabled=1
   gpgcheck=1
   gpgkey=https://repo.charm.sh/yum/gpg.key' | sudo tee /etc/yum.repos.d/charm.repo
   sudo yum install glow
   ```

4. **Usage:**
   To view a Markdown file, run:
   ```bash
   glow your_file.md
   ```
   For a more interactive experience, you can use:
   ```bash
   glow -p your_file.md
   ```

### Conclusion

Both **mdless** and **Glow** are excellent choices for viewing Markdown files in the terminal. Choose the one that best fits your needs and follow the installation instructions above to get started.

Citations:
[1] https://opensource.com/article/20/3/markdown-apps-linux-command-line
[2] https://www.tecmint.com/best-markdown-editors-for-linux/
[3] https://itsfoss.com/glow-cli-tool-markdown/
[4] https://softwarerecs.stackexchange.com/questions/17714/simple-markdown-viewer-for-ubuntu-standalone-program-not-something-that-requir
[5] https://www.reddit.com/r/linux/comments/1023abr/what_is_the_simplest_markdown_viewer/
