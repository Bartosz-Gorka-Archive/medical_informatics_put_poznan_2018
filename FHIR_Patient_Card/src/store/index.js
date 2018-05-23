import Vue from 'vue'
import Vuex from 'vuex'
import createPersistedState from 'vuex-persistedstate'
import patient from './patients/patient'
import medication from './medications/medication'
import observation from './observations/observation'

Vue.use(Vuex)

export default new Vuex.Store({
  modules: {
    patient,
    medication,
    observation
  },
  plugins: [
    createPersistedState()
  ]
})
