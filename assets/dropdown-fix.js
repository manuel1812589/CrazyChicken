document.addEventListener('DOMContentLoaded', function () {
    function fixDropdowns() {
        const dropdowns = document.querySelectorAll('.Select');
        dropdowns.forEach(dropdown => {
            dropdown.style.zIndex = '1000';
            const menu = dropdown.querySelector('.Select-menu-outer');
            if (menu) {
                menu.style.zIndex = '9999';
            }
        });
    }

    fixDropdowns();
    setInterval(fixDropdowns, 1000);

    document.addEventListener('click', function (e) {
        if (e.target.closest('.Select-control')) {
            setTimeout(fixDropdowns, 100);
        }
    });
});