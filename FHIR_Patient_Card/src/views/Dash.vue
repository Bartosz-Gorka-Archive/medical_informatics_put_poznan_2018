<template>
  <div class="l-app">
    <sidebar>
      <div class="logo" slot="start">
        <img width="50" src="/static/images/logo.png" alt="Logo">
      </div>
      <ul class="c-menu c-menu--dark" v-menu slot="content">
        <li class="c-menu__item">
          <router-link :to="{ name: 'patients' }" class="c-menu__link">
            <div class="c-menu__icon">
              <i class="icon-users"></i>
            </div>
            <div class="c-menu__text">Patients</div>
          </router-link>
          <router-link :to="{ name: 'medications' }" class="c-menu__link">
            <div class="c-menu__icon">
              <i class="icon-pill"></i>
            </div>
            <div class="c-menu__text">Medications</div>
          </router-link>
          <router-link :to="{ name: 'observations' }" class="c-menu__link">
            <div class="c-menu__icon">
              <i class="icon-eye"></i>
            </div>
            <div class="c-menu__text">Observations</div>
          </router-link>
        </li>
      </ul>
    </sidebar>

    <div class="l-app__content">
        <div class="l-app__body">
            <router-view></router-view>
        </div>
    </div>
  </div>
</template>

<script>
  export default {
    name: 'DashView'
  }
</script>
