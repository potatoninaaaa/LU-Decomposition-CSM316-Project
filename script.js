document.addEventListener("DOMContentLoaded", function () {
const modal = document.getElementById('sampleModal');
  const openBtn = document.getElementById('openSampleModal');
  const closeBtn = document.getElementById('closeSampleModal');

  openBtn.addEventListener('click', () => {
    modal.classList.add('active');
  });

  closeBtn.addEventListener('click', () => {
    modal.classList.remove('active');
  });

  window.addEventListener('click', (e) => {
    if (e.target === modal) {
      modal.classList.remove('active');
    }
  });
});
