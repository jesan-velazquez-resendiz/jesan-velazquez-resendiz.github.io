(function () {
  function applyIcon() {
    var icon = document.getElementById('theme-icon');
    if (!icon) return;
    var isLight = document.documentElement.classList.contains('light-mode');
    icon.className = isLight ? 'fas fa-moon' : 'fas fa-sun';
  }

  function toggleTheme() {
    var isLight = document.documentElement.classList.toggle('light-mode');
    try { localStorage.setItem('theme', isLight ? 'light' : 'dark'); } catch (e) {}
    applyIcon();
  }

  document.addEventListener('DOMContentLoaded', function () {
    applyIcon();
    var btn = document.getElementById('theme-toggle');
    if (btn) btn.addEventListener('click', toggleTheme);
  });
})();
