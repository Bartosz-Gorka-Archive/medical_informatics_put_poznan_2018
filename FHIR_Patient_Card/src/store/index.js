import Vue from 'vue'
import Vuex from 'vuex'
import createPersistedState from 'vuex-persistedstate'
import patient from './patients/patient'

Vue.use(Vuex)

export default new Vuex.Store({
  modules: {
    patient
  },
  plugins: [
    createPersistedState()
  ]
})
