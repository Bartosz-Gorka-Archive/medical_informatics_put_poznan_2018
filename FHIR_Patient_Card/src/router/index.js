import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router)

function load (component) {
  return () => System.import(`@/${component}.vue`)
}

const router = new Router({
  routes: [
    {
      path: '/',
      component: load('views/Dash'),
      children: [
        {
          path: 'patients',
          name: 'patients',
          component: load('views/Patients')
        },
        {
          path: 'medications',
          name: 'medications',
          component: load('views/Medications')
        }
      ]
    },
    {
      path: '*',
      component: load('views/NotFound')
    }
  ]
})

export default router
