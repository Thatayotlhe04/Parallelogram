/* The page works without JavaScript; this only enhances the copy buttons. */
(() => {
  'use strict';
  const status = document.getElementById('copy-status');
  document.querySelectorAll('[data-copy]').forEach(button => {
    let reset;
    button.addEventListener('click', async () => {
      clearTimeout(reset);
      try {
        await navigator.clipboard.writeText(button.dataset.copy);
        button.classList.add('copied');
        button.querySelector('.copy-label').textContent = 'Copied';
        if (status) status.textContent = 'Copied';
      } catch {
        if (status) status.textContent = 'Could not copy. Select the command and copy it manually.';
      }
      reset = setTimeout(() => {
        button.classList.remove('copied');
        if (status) status.textContent = '';
      }, 2000);
    });
  });
})();
