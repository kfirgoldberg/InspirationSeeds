// Initialize Bulma carousel
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all carousels
    const carousels = bulmaCarousel.attach('.carousel', {
        slidesToScroll: 1,
        slidesToShow: 3,
        infinite: true,
        autoplay: true,
        autoplaySpeed: 3000,
        pagination: false
    });

    // Initialize all sliders
    const sliders = bulmaSlider.attach();
});
