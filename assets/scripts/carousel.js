// Initialize all carousels on page load
document.addEventListener('DOMContentLoaded', function() {
    initAllCarousels();
});

function initAllCarousels() {
    const carousels = document.querySelectorAll('.carousel');
    carousels.forEach(carousel => {
        initCarousel(carousel);
    });
}

function initCarousel(carousel) {
    const dots = carousel.querySelectorAll('.dot');
    const slides = carousel.querySelectorAll('.slide-content');

    if (dots.length === 0 || slides.length === 0) return;

    dots.forEach((dot, index) => {
        dot.addEventListener('click', () => {
            // Remove active class from all dots and slides
            dots.forEach(d => d.classList.remove('active'));
            slides.forEach(s => s.classList.remove('active'));

            // Add active class to clicked dot and corresponding slide
            dot.classList.add('active');
            if (slides[index]) {
                slides[index].classList.add('active');
            }
        });
    });
}
