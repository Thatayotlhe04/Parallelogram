/* The page works without JavaScript; enhance only the install copy buttons. */
(() => {
  'use strict';
  if (!navigator.clipboard?.writeText) return;

  const status = document.getElementById('copy-status');
  document.querySelectorAll('[data-copy]').forEach(button => {
    button.hidden = false;
    let reset;
    button.addEventListener('click', async () => {
      clearTimeout(reset);
      status.textContent = '';
      try {
        await navigator.clipboard.writeText(button.dataset.copy);
        button.textContent = 'Copied';
        button.setAttribute('aria-label', 'Copied install command');
        status.textContent = 'Copied';
      } catch {
        button.textContent = 'Retry';
        button.setAttribute('aria-label', 'Retry copying install command');
        status.textContent = 'Could not copy. Select the install command and copy it manually.';
      }
      reset = setTimeout(() => {
        button.textContent = 'Copy';
        button.setAttribute('aria-label', 'Copy install command');
      }, 2500);
    });
  });
})();
